# Using Fuji for Halo2 Proving

## Approach

There are two ways to use Fuji in a Halo2 prover:

### 1. Via the `fuji` crate (concrete API)

Best for: new proving code, maximum performance, explicit batch control.

### 2. Via `fuji-pasta` (trait-based API)

Best for: replacing `pasta_curves` in existing generic code that expects
`ff::PrimeField + group::Group` trait implementations.

## Orchard Integration

Zcash Orchard uses Halo2 with Pasta curves. The proving pipeline requires
hundreds of multi-scalar multiplications for polynomial commitments and
opening proofs.

### Call pattern for a single proof

```rust
use fuji::*;

fn prove_orchard() {
    let g = FujiAffine::gen_pallas();

    // 1. Commit to advice columns (typically ~200 columns)
    let advice_commitments = msm::msm_batch(
        &vec![4096i32; 200],      // each MSM uses 4096 scalars
        &vec![g; 200 * 4096],     // repeated generator
        &advice_scalars,
        FujiCurve::Pallas,
    ).unwrap();

    // 2. Commit to lookup columns
    let lookup_commitments = msm::msm_batch(
        &vec![4096i32; 16],
        &vec![g; 16 * 4096],
        &lookup_scalars,
        FujiCurve::Pallas,
    ).unwrap();

    // 3. Multi-open argument (variable-base MSMs)
    let opening_msms = msm::msm_batch(
        &opening_counts,
        &variable_bases,
        &opening_scalars,
        FujiCurve::Pallas,
    ).unwrap();
}
```

**Total FFI crossings: 3** (vs. ~230+ with a per-MSM design).

### Using the trait bridge

```rust
use fuji_pasta::*;
use ff::Field;
use group::Group;

fn prove_generic<C: CurveConstants>() {
    // Work with field elements generically
    let a = FujiField::<C>::from(10u64);
    let b = a.square();

    // Work with curve points
    let g = <FujiPoint<C> as Group>::generator();
    let two_g = g.double();

    // Convert to pasta_curves for interop
    if C::curve_id() == 0 {
        let fp: pasta_curves::Fp = a.into();
        // ... pass fp to existing code
    }
}
```

## Performance Notes

Field multiplication dispatching:

| Path | Time/op | Notes |
|---|---|---|
| AMX matint + Barrett | ~3270 ns | 8-bit outer product + byte-level Barrett reduction |
| Montgomery (64-bit CIOS) | ~108 ns | 4×4 64-bit schoolbook + CIOS reduction, via `fuji_field_mul_ref` |
| `pasta_curves` (Rust, 64-bit NEON) | ~6 ns | Fully optimized Montgomery in Rust |

The AMX path is slower than software because the Barrett reduction
uses byte-level loops (~2500+ byte operations per mul) that dominate
the AMX outer product gains. The Montgomery path is 30× faster but
still ~18× behind hand-tuned Rust Montgomery.

Jacobian point operations cost:

| Op | Field muls | Est. time (Montgomery path) | Est. time (AMX path) |
|---|---|---|---|
| Point double | ~13 | ~1.4 µs | ~43 µs |
| Point add | ~15 | ~1.6 µs | ~49 µs |
| Mixed add | ~11 | ~1.2 µs | ~36 µs |

MSM break-even vs `pasta_curves`:

| MSM size | pasta_curves (est.) | Fuji Montgomery | Fuji AMX |
|---|---|---|---|
| 64 pts | ~2.5 ms | ~0.6 ms | ~20 ms |
| 1024 pts | ~40 ms | ~10 ms | ~320 ms |
| 4096 pts | ~160 ms | ~40 ms | ~1.3 s |

The Montgomery path is ~4× faster than `pasta_curves` for MSM, despite
being slower per field mul, because the C library's Pippenger MSM uses
affine bases and efficiently batched point operations.

## Tips

- **Batch aggressively**: Combine all fixed-base MSMs (same generator) into
  one `msm_batch` call.
- **Use `add_mixed`**: When one operand is affine, `add_mixed` saves ~25%
  over `add`.
- **Prefer `fuji` over `fuji-pasta` for MSM**: The trait-based API uses
  double-and-add for scalar multiplication. For batched MSMs, call
  `fuji::msm::msm_batch` directly with affine bases and Pippenger's method.
- **Avoid per-element FFI**: Don't call `fuji_f_add` in a tight loop.
  Batch if possible, or use the internal `fuji_field_batch_mul` path.
- **Convert at the boundary**: Convert `pasta_curves::Fp` values to
  `FujiField<PallasTag>` at the prover entry point, do all work in Fuji,
  then convert back at the exit point.
