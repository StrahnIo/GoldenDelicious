# Architecture

## Crate Layering

```
┌────────────────────────────────────────────────────────────────┐
│                    Halo2 / orchard / custom prover               │
│  (generic over ff::PrimeField + group::Group)                   │
└──────────────────────┬─────────────────────────────────────────┘
                       │ trait-based API
┌──────────────────────▼─────────────────────────────────────────┐
│              fuji-pasta (trait bridge)                          │
│  FujiField<C>: ff::PrimeField, ff::Field                       │
│  FujiPoint<C>: group::Group                                    │
│  FujiField<PallasTag> ↔ pasta_curves::Fp                       │
└──────────────────────┬─────────────────────────────────────────┘
                       │ concrete types + error codes
┌──────────────────────▼─────────────────────────────────────────┐
│              fuji (safe Rust wrappers)                          │
│  FujiField, FujiPoint, FujiAffine                              │
│  msm_eval, msm_batch                                           │
│  Result<_, FujiError>                                          │
└──────────────────────┬─────────────────────────────────────────┘
                       │ extern "C" (unsafe)
┌──────────────────────▼─────────────────────────────────────────┐
│              fuji-sys (raw FFI bindings)                        │
│  #[link(name = "fuji")] extern "C" { ~40 fns }                 │
│  #[repr(C)] fuji_field, fuji_affine, fuji_point                │
└──────────────────────┬─────────────────────────────────────────┘
                       │ dynamic linking (libfuji.dylib)
┌──────────────────────▼─────────────────────────────────────────┐
│              libfuji.dylib (C library)                          │
│  AMX matint, Montgomery reduction, Pippenger MSM               │
│  SME UMOPA outer product (M4+)                                 │
│  No stdlib dependency, only macOS + Apple Silicon              │
└────────────────────────────────────────────────────────────────┘
```

## Dispatch Order

`fuji_field_mul` selects the multiplication path in this order:

1. **Montgomery** (software 64-bit) — 4×4 64-bit schoolbook + CIOS
   Montgomery reduction. ~116 ns per mul (dispatch path).
2. **AMX** (Apple Matrix Extensions) — 8-bit `matint` outer product +
   Barrett reduction. Undocumented, ~25× slower than Montgomery.

For batch multiplication, the **PRL** extension provides 2-3× interleaved
CIOS via `mont_mul_2x` (~17 ns/mul) and `mont_mul_3x` (~15.7 ns/mul).

SME (Scalable Matrix Extension) UMOPA outer product compiles on M4
but is not dispatched due to toolchain limitations (LLVM assembler
does not support ZA tile extraction instructions).

## The AMX Coprocessor

Apple's AMX (Apple Matrix Extensions) is an undocumented matrix coprocessor on
every Apple Silicon SoC. It executes 23 instruction types including:

- **FMA32/FMA64**: 32/64-bit float fused multiply-add (outer product)
- **MATINT**: Integer matrix multiply-accumulate (8/16-bit)
- **VECINT**: Vector integer operations

Fuji uses `MATINT` with `alumode=8`, 8-bit data type, to compute the
32×32 outer product of two field elements in a single instruction.

Key properties:
- **Per-cluster**, not per-core (one AMX serves all P-cores in a cluster)
- **1 cycle throughput** for independent operations (P-core AMX)
- **4 cycle latency** — pipeline with independent Z row offsets
- Must be explicitly enabled/disabled via `AMX_SET()` / `AMX_CLR()`

## Field Multiplication Pipeline

### AMX path (legacy, ~3269 ns)

```
a (32 bytes) ──┐                    ┌── b (32 bytes)
               ▼                    ▼
        ┌──────────────┐   ┌──────────────┐
        │ Pad to 64 B  │   │ Interleaved  │
        └──────┬───────┘   └──────┬───────┘
               │                  │
               ▼                  ▼
        ┌─────────────────────────────┐
        │  AMX_MATINT(alumode=8)      │  outer product
        │  Z[j][i] += X[i] * Y[j]    │  1024 partial products
        │  i8 × i8 → i16              │
        └─────────────────────────────┘
                      │
               ┌──────▼──────┐
               │  Barrett     │
               │  reduction   │  q ≈ t/p, r = t - q*p
               │  csub loop   │  up to 10 iterations
               └──────▼──────┘
                      │
                   a·b mod p
```

### PRL path (parallel, ~16-17 ns per mul)

```
              ┌─── a0,b0 ───┐  ┌─── a1,b1 ───┐  ┌─── a2,b2 ───┐
              ▼              ▼  ▼              ▼  ▼              ▼
         ┌─────────────────────────────────────────────────────────┐
         │            3× interleaved CIOS Montgomery              │
         │   (mont_mul_3x — 3 streams in lockstep)                │
         │   Schoolbook: 16 MACs × 3 streams = 48 MACs            │
         │   CIOS:        4 iters × 4 steps × 3 streams = 48 MACs │
         └─────────────────────────────────────────────────────────┘
              │              │  │              │  │              │
              ▼              ▼  ▼              ▼  ▼              ▼
           r0 = a0·b0      r1 = a1·b1      r2 = a2·b2
              mod p          mod p            mod p
```

The PRL extension exploits instruction-level parallelism across independent
field multiplications. By interleaving 2-3 streams' MUL/UMULH/ADCS sequences,
the M4's 4-wide pipelines stay filled.

| Variant | Streams | ns/quad | ns/mul | Speedup |
|---------|---------|---------|--------|---------|
| Sequential | 1 | — | ~22 | 1× |
| `mont_mul_2x` | 2 | ~34 | **~17** | 1.3× |
| `mont_mul_3x` | 3 | ~47 | **~15.7** | 1.4× |

### Montgomery path (primary, ~108 ns)

```
a ──┐               ┌── b
    ▼               ▼
┌─────────────────────────┐
│  4×4 64-bit schoolbook  │  16 mul + carry → 8× uint64_t
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  CIOS reduction          │  4 iterations × 4 mul-add + carry
│  (combined q*R⁻¹ mod p)  │
└──────────┬──────────────┘
           ▼
┌─────────────────────────┐
│  Conditional subtract   │  if r ≥ p: r -= p
└──────────▼──────────────┘
           │
        a·b mod p
```

## MSM (Pippenger's Bucket Method)

For `Σ sᵢ · Pᵢ` with 255-bit scalars:

1. **Window**: 8-bit windows → 32 windows per scalar
2. **Bucket**: For each window j, 256 buckets `B[0..255]`
3. **Accumulate**: For each pair `(sᵢ, Pᵢ)`, add `Pᵢ` to `B[windowⱼ(sᵢ)]`
4. **Reduce**: `running = Σˢ B[v]`, `acc = Σ running`
5. **Combine**: `result = Σⱼ accⱼ · 2^(8·j)`

Total group ops per MSM: approximately `n + 256` additions and `log₂(n)·255`
doublings.

### Pure-PRL Pippenger (Mont-form, no Barrett)

`fuji::msm::prl_pippenger` implements Pippenger entirely in Montgomery form
using `prl::mul_3x` for every field operation. No Mont↔normal conversions
during fill, reduction, or combination. Key costs:

| Phase | Ops | Ops used | ns/op | Total |
|-------|-----|----------|-------|-------|
| Bucket fill (PRL) | `n` adds | `prl_add_mixed_3` | 176ns/3 | ~59n·ns |
| Bucket fill (seq) | `n` adds | `mont_mixed_add` | 96ns | ~96n·ns |
| Bucket reduction | 8160 adds | `mont_add` | 112ns | ~914µs |
| Window combination | 288 ops | `mont_double`/`mont_add` | 32-112ns | ~15µs |

The Mont-form point operations are written in pure Rust using `prl::mul_3x`
and `f_sub`/`f_add` (C library field add/sub — ring homomorphism, preserves
Mont form).

| Function | Description | mul_3x calls | ns |
|----------|-------------|-------------|----|
| `mont_double` | Jacobian doubling (a=0) | 3 | ~48 |
| `mont_mixed_add` | Jacobian + affine | 7 | ~112 |
| `mont_add` | Jacobian + Jacobian | 7 | ~112 |

## FFI Boundary

Data crossing the Rust→C boundary:

| Data | Direction | Size | Frequency |
|---|---|---|---|
| `fuji_field` | Rust → C | 32 bytes | Per field op |
| `fuji_field` | C → Rust | 32 bytes | Per field op |
| `fuji_affine` | Rust → C | 64 bytes | Per MSM base |
| `fuji_point` | C → Rust | 96 bytes | Per MSM result |
| error code | C → Rust | 4 bytes | Per operation |

The batch MSM interface (`fuji_msm_batch`) reduces the number of FFI
crossings from O(n_msms) to O(1) by concatenating all inputs.

## Type-Level Curve Tracking

`fuji-pasta` uses phantom type parameters to prevent curve confusion:

```
FujiField<PallasTag>  — Arithmetic in Fp (Pallas base field)
FujiField<VestaTag>   — Arithmetic in Fq (Vesta base field)
FujiPoint<PallasTag>  — Points on the Pallas curve
FujiPoint<VestaTag>   — Points on the Vesta curve
```

Scalar multiplication on the Pallas curve uses `FujiField<VestaTag>`
(because the Pallas scalar field is Fq, which is Vesta's base field).
This is enforced at compile time — mismatched fields cause type errors.
