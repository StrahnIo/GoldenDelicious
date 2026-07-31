# Fuji Rust Documentation

Fuji is a Rust FFI crate for Apple Silicon providing hardware-accelerated
elliptic curve arithmetic for Halo2 zero-knowledge proofs.

**Performance:**
- `fuji_field_mul`: **~116 ns** (Montgomery CIOS, dispatched by default)
- `mont_mul_3x` (PRL): **~16 ns/mul** when batching 3 independent muls
- 2× interleaved CIOS also available via PRL extension

## Contents

| Document | Description |
|---|---|
| [Build](build.md) | Compiling from source, linking `libfuji.dylib` |
| [API Reference](api.md) | Safe Rust API surface across all crates |
| [Architecture](architecture.md) | How Fuji works under the hood |
| [Trait Bridge](trait-bridge.md) | `ff`/`group` trait implementations in `fuji-pasta` |
| [Halo2 Proving](proving.md) | Using Fuji for Orchard and custom circuits |

## Quick Start

### Using the safe API (`fuji` crate)

```rust
use fuji::*;

// Check AMX is available
assert!(detection::amx_available());

// SME availability (M4 only)
let has_sme = detection::sme_available();

// PRL (parallel) — always available, batch field mul
assert!(prl::prl_available());
let (aa, _, _) = prl::mul_3x(&one, &one, &one, &one, &one, &one, FujiCurve::Pallas).unwrap();

// Generator for the Pallas curve
let g = FujiAffine::gen_pallas();

// Field arithmetic
let one = FujiField::one();
let two = FujiField::from_bytes(&[2u8; 32]);

// Multi-scalar multiplication
let result = msm::msm_eval(&[g, g], &[one, two], FujiCurve::Pallas).unwrap();
println!("1*G + 2*G = {:02x?}", result.x_limbs());

// Cleanup signal handlers before exit
fuji::cleanup();
```

### Using the trait bridge (`fuji-pasta` crate)

```rust
use fuji_pasta::*;
use ff::Field;
use group::Group;

// FujiField implements ff::PrimeField + ff::Field
let a = FujiField::<PallasTag>::from(10u64);
let b = a.double();
assert_eq!(b, a + a);

// FujiPoint implements group::Group
let g = <FujiPoint<PallasTag> as Group>::generator();
let two_g = g.double();
assert!(!bool::from(two_g.is_identity()));

// Convert to/from pasta_curves types
let fp: pasta_curves::Fp = a.into();
let back: FujiField<PallasTag> = fp.into();
assert_eq!(a, back);
```

## Project Structure

```
rust/
├── Cargo.toml              # Workspace root
├── fuji-sys/               # Raw extern "C" FFI declarations
│   ├── build.rs            # Locates libfuji.dylib
│   └── src/lib.rs          # ~40 extern C functions, #[repr(C)] types
├── fuji/                   # Safe Rust wrappers
│   ├── src/lib.rs          # Crate root, FujiCurve enum
│   ├── src/field.rs        # FujiField — 255-bit Pasta field ops
│   ├── src/curve.rs        # FujiAffine, FujiPoint — curve ops
│   ├── src/msm.rs          # Multi-scalar multiplication
│   ├── src/detection.rs    # AMX/SME/PRL detection, CPU info
│   ├── src/prl.rs          # PRL parallel batch multiplication
│   └── src/error.rs        # FujiError enum
├── fuji-pasta/             # group/ff/PrimeField trait impls
│   ├── src/lib.rs          # FujiField<C>, FujiPoint<C> tagged wrappers
│   └── tests/bridge.rs     # 7 integration tests
└── docs/                   # This documentation
```

## Crates

### `fuji-sys`

Zero-dependency crate with raw `extern "C"` declarations for all
libfuji functions (~40 functions). All functions are `unsafe`.
See `fuji` for the safe API.

### `fuji`

Safe Rust wrappers with `Result`-based error handling, `Copy`/`Clone`/`Debug`
derives, and `PartialEq`/`Eq` for field elements.

### `fuji-pasta`

Type-safe tagged wrappers `FujiField<C>`, `FujiPoint<C>`, `FujiAffine<C>` with
a `CurveTag` phantom parameter that tracks the curve at the type level.
Implements `ff::PrimeField`, `ff::Field`, and `group::Group` traits for
interop with `pasta_curves` and generic Halo2 code.

## License

MIT
