# Trait Bridge (`fuji-pasta`)

The `fuji-pasta` crate bridges Fuji's concrete types with the standard
`ff` and `group` Rust traits, enabling Fuji to slot into generic Halo2 code
that expects `PrimeField`, `Field`, and `Group` implementations.

## Motivation

The `fuji` crate provides concrete types (`FujiField`, `FujiPoint`) that
require an explicit `FujiCurve` parameter for every arithmetic operation:

```rust
// fuji crate: curve must be specified each time
let result = a.mul(&b, FujiCurve::Pallas).unwrap();
```

Halo2 and other generic proving code use trait-based abstractions where the
curve is encoded at the type level. `fuji-pasta` provides tagged wrappers
that implement these traits.

## Approach

`fuji-pasta` uses a phantom type parameter `C: CurveTag` to encode the curve:

```rust
#[repr(transparent)]
pub struct FujiField<C: CurveTag>(fuji::FujiField, PhantomData<C>);
```

The `CurveTag` trait has two implementations:

| Marker | `curve_id()` | Curve |
|---|---|---|
| `PallasTag` | `0` | Pallas (base = Fp, scalar = Fq) |
| `VestaTag` | `1` | Vesta (base = Fq, scalar = Fp) |

Because the Pasta curves form a 2-cycle, a second trait associates each
curve tag with its scalar field tag:

```rust
trait CurveScalar: CurveTag { type Scalar: CurveTag; }
// PallasTag::Scalar = VestaTag
// VestaTag::Scalar  = PallasTag
```

## Type Safety

The phantom parameter prevents mixing field elements from different curves:

```rust
// Compile error: mismatched types
let fp = FujiField::<PallasTag>::from(1u64);
let fq = FujiField::<VestaTag>::from(2u64);
let sum = fp + fq;  // ERROR
```

Cross-curve operations are explicitly permitted where they make sense —
for example, multiplying a `FujiPoint<PallasTag>` by a `FujiField<VestaTag>`
(the Pallas scalar field is `Fq`, which is `VestaTag`'s base field).

## Trait Implementations

### `ff::Field` for `FujiField<C>`

| Item | Value |
|---|---|
| `ZERO` | Additive identity |
| `ONE` | Multiplicative identity |
| `random(rng)` | Random field element |
| `is_zero()` | Constant-time zero check |
| `double()` | `self + self` |
| `square()` | `self * self` |
| `invert()` | Multiplicative inverse (`CtOption`) |
| `sqrt()` | Square root (`CtOption`) |
| `sqrt_ratio()` | Combined sqrt ratio |

### `ff::PrimeField` for `FujiField<C>`

| Item | Value |
|---|---|
| `Repr = [u8; 32]` | Canonical byte representation |
| `NUM_BITS` | `255` |
| `CAPACITY` | `254` |
| `S` | `32` (2-adicity of both Pasta moduli) |
| `TWO_INV` | `(p+1)/2` (per curve constant) |
| `from_repr(bytes)` | Constant-time canonical check |
| `to_repr()` | Canonical bytes |
| `is_odd()` | LSB of `limbs[0]` |

### `group::Group` for `FujiPoint<C>`

| Item | Value |
|---|---|
| `Scalar` | `FujiField<C::Scalar>` (the scalar field for this curve) |
| `identity()` | Point at infinity |
| `generator()` | `(-1, 2)` for the curve |
| `random(rng)` | `r * G` |
| `is_identity()` | `Choice` wrapping projective Z-check |
| `double()` | Jacobian doubling (AMX-accelerated) |

Scalar multiplication is implemented via double-and-add over the 255-bit
scalar, using the `fuji` crate's projective point operations.

### Additional methods on `FujiPoint<C>`

| Method | Description |
|---|---|
| `p.to_affine() -> FujiAffine` | Jacobian → affine conversion (uses type-level curve) |
| `FujiPoint::<C>::batch_to_affine(&[Self]) -> Vec<FujiAffine>` | Batch conversion with batched inverses |
| `p.neg()` | Point negation |

## `pasta_curves` Interop

`From` conversions are provided between `FujiField<PallasTag>` and
`pasta_curves::Fp`, and between `FujiField<VestaTag>` and `pasta_curves::Fq`.
This lets you wrap existing `pasta_curves` values, accelerate MSM with Fuji,
and unwrap the results back.

```rust
use fuji_pasta::*;
use ff::Field;

// Convert pasta_curves scalar to Fuji
let fp = pasta_curves::Fp::from(42u64);
let fuji_fp: FujiField<PallasTag> = fp.into();

// Use Fuji's AMX-accelerated arithmetic
let result = fuji_fp.square();

// Convert back
let back: pasta_curves::Fp = result.into();
```

## Testing

```shell
cd rust
DYLD_LIBRARY_PATH=../.. cargo test -p fuji-pasta -- --test-threads=1
```

7 integration tests cover: field round-trips (Pallas, Vesta), field add/mul,
group generator, scalar multiplication, and ff::Field basics.

## Performance Note

Scalar multiplication via double-and-add (the `*` operator on `FujiPoint`) is
not AMX-accelerated at the MSM level — it uses projective point operations
which each call AMX field multiplies. For optimal performance, use
`fuji::msm::msm_eval` or `fuji::msm::msm_batch` directly with affine bases.
