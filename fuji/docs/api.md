# API Reference

## `fuji` crate (safe wrappers)

---

### `fuji::FujiCurve`

```rust
pub enum FujiCurve { Pallas = 0, Vesta = 1 }
```

Selects the Pasta curve. `Pallas` is the main curve for Orchard;
`Vesta` is used for the recursive proof layer.

### `fuji::FujiField`

255-bit field element over either Pallas or Vesta base field.

| Method | Description |
|---|---|
| `FujiField::zero()` | Additive identity |
| `FujiField::one()` | Multiplicative identity |
| `FujiField::from_bytes(&[u8; 32])` | Load from little-endian bytes |
| `FujiField::to_bytes(&self) -> [u8; 32]` | Store as little-endian bytes |
| `a.add(&b, curve) -> Result` | `a + b mod p` |
| `a.sub(&b, curve) -> Result` | `a - b mod p` |
| `a.mul(&b, curve) -> Result` | `a * b mod p` (AMX-accelerated) |
| `a.sqr(curve) -> Result` | `a² mod p` |
| `a.inv(curve) -> Result` | `a⁻¹ mod p` (Fermat) |
| `a.sqrt(curve) -> Result` | `√a mod p` (Tonelli-Shanks) |
| `a.eq(&b) -> bool` | Constant-time equality |

All arithmetic methods take `curve: FujiCurve` to select the modulus.

### `fuji::FujiAffine`

Affine point `(x, y)` on a Pasta curve.

| Method | Description |
|---|---|
| `FujiAffine::gen_pallas()` | Pallas generator `(-1, 2)` |
| `FujiAffine::gen_vesta()` | Vesta generator `(-1, 2)` |
| `FujiAffine::from_coordinates(x, y)` | Construct from field elements |
| `a.x()` | Reference to x-coordinate |
| `a.y()` | Reference to y-coordinate |

### `fuji::FujiPoint`

Jacobian projective point `(X, Y, Z)` on a Pasta curve.

| Method | Description |
|---|---|
| `FujiPoint::identity()` | Point at infinity |
| `p.is_identity() -> bool` | `true` if `Z == 0` |
| `FujiPoint::from_affine(&a, curve)` | Convert affine → Jacobian |
| `p.double(curve) -> Result` | `2·p` |
| `p.add(&q, curve) -> Result` | `p + q` |
| `p.add_mixed(&a, curve) -> Result` | `p + a` (mixed, ~25% faster) |
| `p.negate(curve) -> Result` | `-p` (negates Y) |
| `p.to_affine(curve) -> Result<FujiAffine>` | Jacobian → affine |
| `p.to_affine_unchecked(curve) -> FujiAffine` | Panics on error |
| `FujiPoint::batch_to_affine(&[FujiPoint], curve) -> Result<Vec<FujiAffine>>` | Batch conversion |
| `p.x_limbs() -> &[u8; 32]` | X-coordinate bytes |
| `p.y_limbs() -> &[u8; 32]` | Y-coordinate bytes |
| `p.z_limbs() -> &[u8; 32]` | Z-coordinate bytes |

**Breaking change:** `to_affine` now requires a `curve` parameter. Previous
no-parameter version always used PALLAS modulus.

### `fuji::msm`

| Function | Description |
|---|---|
| `msm::msm_eval(&[FujiAffine], &[FujiField], curve) -> Result` | Single MSM |
| `msm::msm_batch(&[i32], &[FujiAffine], &[FujiField], curve) -> Result<Vec<...>>` | Batch MSM |

### `fuji::detection`

| Function | Description |
|---|---|
| `detection::amx_available() -> bool` | `true` if AMX coprocessor present |
| `detection::sme_available() -> bool` | `true` if SME (M4+) present |
| `detection::cpu_brand() -> &'static str` | e.g. `"Apple M4"` |
| `detection::lib_version() -> &'static str` | e.g. `"0.1.0"` |

### `fuji::prl`

Batch field multiplication via the PRL (parallel) extension. Processes 2–3
independent field multiplications simultaneously using interleaved CIOS,
filling the M4's 4-wide pipelines.

**All inputs and outputs must be in Montgomery form.** The PRL API operates
directly on Montgomery-form values — no conversion overhead. If you have
normal-form values, convert them once at batch boundaries.

**Montgomery form conversion** (all implemented in C for performance):

```rust
// Using FujiField (recommended — single FFI call, ~116 ns):
let a_mont = a.to_mont(FujiCurve::Pallas);
let a_back = a_mont.from_mont(FujiCurve::Pallas);

// Using pasta_curves::Fp (pure Rust, slower):
let fp = pasta_curves::Fp::from_raw([...]);
let r2 = pasta_curves::Fp::from_raw([0x8c78ecb30000000f, 0xd7d30dbd8b0de0e7,
                                      0x7797a99bc3c95d18, 0x096d41af7b9cb714]);
let fp_mont = fp * r2;  // = fp * R mod p
```

| Function | Description |
|---|---|
| `prl::prl_available() -> bool` | `true` (software feature, always available) |
| `prl::mul_2x(a0,b0, a1,b1, curve) -> Result<(Fp,Fp)>` | 2× parallel mul (~17 ns/mul) |
| `prl::mul_3x(a0,b0, a1,b1, a2,b2, curve) -> Result<(Fp,Fp,Fp)>` | 3× parallel mul (~16 ns/mul) |
| `curve::mont_double(a, curve) -> Result<FujiPoint>` | Mont-form Jacobian doubling (~48ns) |
| `curve::mont_add(a, b, curve) -> Result<FujiPoint>` | Mont-form Jacobian addition (~112ns) |
| `curve::mont_mixed_add(a, b, curve) -> Result<FujiPoint>` | Mont-form mixed addition (~112ns) |
| `msm::prl_pippenger(scalars, bases, curve) -> Result<FujiPoint>` | Mont-form Pippenger (pure PRL) |
| `msm::prl_msm_batch(counts, bases, scalars, curve) -> Result<Vec<FujiPoint>>` | Batch MSM via prl_pippenger |

**Performance guidance:**
- Use `mul_3x` when you have ≥3 independent multiplications to batch (e.g.,
  processing field elements from 3 different MSM buckets, or 3 independent
  proof statements). Throughput: **~16 ns/mul** vs ~116 ns for sequential.
- Use `mul_2x` for pairs. Throughput: **~17 ns/mul**.
- Avoid converting to/from Montgomery form per-call — batch conversions at
  the boundary of the batch operation.
- `prl_pippenger` uses zero Montgomery conversions during fill/reduce/combine.
  Keep bases in Mont form; convert result with `.from_mont()`.
- `mont_add`, `mont_double`, `mont_mixed_add` operate entirely in Montgomery
  form using `prl::mul_3x`. They never call the C library's `fuji_field_mul`
  (which goes through Barrett reduction).

**Pipeline diagram:**
```
Normal form  ─→  to_mont(x R2)  ─→  Mont form  ─→  PRL batch  ─→  Mont form  ─→  from_mont(x 1)  ─→  Normal
  (slow)                          (fast)                       (fast)                            (slow)
```
Keep values in Montgomery form across PRL batches for maximum throughput.

### `fuji::sme`

Low-level SME (Scalable Matrix Extension) primitives for the M4's ZA matrix tile.

| Function / Type | Description |
|---|---|
| `SmeStream::enter()` | Enter streaming SVE/SME mode. Returns a RAII guard that exits on drop. Panics if SME unavailable. |
| `SmeStream::is_active() -> bool` | `true` if a streaming guard is alive on this thread. |
| `sme::umopa_outer_product(a: &[u8;32], b: &[u8;32]) -> [u32; 1024]` | Compute 32×32 8-bit unsigned outer product via `UMOPA`. Internal STR ZA extraction may `SIGILL` on some toolchains. |
| `sme::umopa_outer_product_raw(a, b, out) — unsafe` | Raw variant with caller-provided buffer. |

### `fuji::cleanup()`

Restores signal handlers (SIGILL for AMX detection, SIGILL for SME ZA).
Safe to call multiple times; idempotent.

### `fuji::FujiError`

| Variant | Meaning |
|---|---|
| `AmxUnavailable` | AMX not present; library requires AMX |
| `InvalidContext` | NULL pointer or uninitialised state |
| `OutOfMemory` | `calloc` failed |
| `Generic` | Unknown error |

Converts from C error codes via `From<i32>`.

---

## `fuji-pasta` crate (trait bridge)

---

### `fuji_pasta::CurveTag`

```rust
pub trait CurveTag: Copy + Clone + Send + Sync + 'static {
    fn curve_id() -> i32;
}
```

Zero-sized marker types that encode the curve at the type level.

| Tag | Curve | `curve_id()` |
|---|---|---|
| `PallasTag` | Pallas | `0` |
| `VestaTag` | Vesta | `1` |

### `fuji_pasta::CurveScalar`

Associates the scalar field for a curve (Pasta cycle property):

| Curve Tag | Scalar Tag |
|---|---|
| `PallasTag` | `VestaTag` |
| `VestaTag` | `PallasTag` |

### `fuji_pasta::CurveConstants`

Per-curve field constants for `ff::PrimeField`:

| Constant | Description |
|---|---|
| `MODULUS_STR` | Hex modulus string |
| `MODULUS_BYTES` | 32-byte LE modulus |
| `TWO_INV_BYTES` | `(p+1)/2` as 32-byte LE |

### `fuji_pasta::FujiField<C>`

Newtype over `fuji::FujiField` with a `CurveTag` phantom parameter.
Implements `ff::PrimeField`, `ff::Field`, `Add`, `Sub`, `Mul`, `Neg`,
`AddAssign`, `SubAssign`, `MulAssign`, `Sum`, `Product`, `From<u64>`.

| Method | Description |
|---|---|
| `FujiField::<C>::zero()` | Additive identity |
| `FujiField::<C>::one()` | Multiplicative identity |
| `FujiField::<C>::from_bytes(&[u8;32])` | Construct from bytes |
| `f.to_bytes() -> [u8; 32]` | Canonical bytes |
| `a + b`, `a - b`, `a * b` | Arithmetic via operators (no curve parameter needed — it's in the type) |
| `a.double()`, `a.square()` | Specialised operations |
| `a.neg()` | Additive inverse |
| `a.invert() -> CtOption` | Multiplicative inverse |
| `a.sqrt() -> CtOption` | Square root |
| `a.is_zero() -> Choice` | Constant-time zero check |
| `a.is_odd() -> Choice` | LSB check |

### `fuji_pasta::FujiAffine<C>` / `FujiPoint<C>`

Newtypes over `fuji::FujiAffine`/`fuji::FujiPoint` with a `CurveTag` phantom.

| Method | Description |
|---|---|
| `FujiPoint::<C>::identity()` | Point at infinity |
| `p.is_identity() -> bool` | `true` if `Z == 0` |
| `FujiPoint::<C>::from_affine(&a)` | Affine → Jacobian |
| `p.double()` | `2·p` |
| `p + q` | Point addition via `+` operator |
| `p * s` | Scalar multiplication via `*` operator |
| `p.neg()` | Negation |
| `p.to_affine() -> FujiAffine` | Jacobian → affine (curve from type) |
| `FujiPoint::<C>::batch_to_affine(&[Self]) -> Vec<FujiAffine>` | Batch conversion |
| `g * s` (where `g: FujiPoint<PallasTag>`, `s: FujiField<VestaTag>`) | Cross-curve scalar mul |

### trait implementations on `FujiPoint<C>`

| Trait | Provides |
|---|---|
| `group::Group` | `identity()`, `generator()`, `random()`, `is_identity()`, `double()` |
| `Add`, `Sub`, `Neg`, `AddAssign`, `SubAssign` | Group operations |
| `Sum` | Summation over iterators |
| `Mul`, `MulAssign` for scalar multiplication | Double-and-add over the 255-bit scalar |

### `pasta_curves` conversions

| Source | Target | Method |
|---|---|---|
| `FujiField<PallasTag>` | `pasta_curves::Fp` | `.into()` |
| `pasta_curves::Fp` | `FujiField<PallasTag>` | `.into()` |
| `FujiField<VestaTag>` | `pasta_curves::Fq` | `.into()` |
| `pasta_curves::Fq` | `FujiField<VestaTag>` | `.into()` |
