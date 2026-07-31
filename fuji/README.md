# Fuji Rust

Safe Rust bindings for [libfuji](https://github.com/anomalyco/fuji) — AMX-accelerated
elliptic curve arithmetic for Halo2 proving on Apple Silicon.

## Workspace

| Crate | Description |
|---|---|
| `fuji-sys` | Raw `extern "C"` FFI declarations (~40 fns). Zero dependencies. |
| `fuji` | Safe Rust API: `FujiField`, `FujiPoint`, `FujiAffine`, MSM. |
| `fuji-pasta` | `ff::PrimeField` + `group::Group` trait impls. Interop with `pasta_curves`. |

## Documentation

See the [`docs/`](docs/index.md) directory for full documentation:

| Document | Description |
|---|---|
| [Build](docs/build.md) | Compiling from source, troubleshooting |
| [API Reference](docs/api.md) | Complete API surface (fuji + fuji-pasta) |
| [Architecture](docs/architecture.md) | AMX pipeline, Montgomery reduction, MSM |
| [Trait Bridge](docs/trait-bridge.md) | `ff`/`group` trait implementations |
| [Halo2 Proving](docs/proving.md) | Integration guide for Orchard |

## Build

```shell
# Build the C library first
make -C ..

# Then build Rust crates
cd rust
DYLD_LIBRARY_PATH=../.. cargo build --release
```

## Test

```shell
cd rust
DYLD_LIBRARY_PATH=../.. cargo test -- --test-threads=1
```

Single-threaded execution is required because the AMX coprocessor is
shared per CPU cluster; Rust's default parallel testing causes crashes.

## Quick start

```rust
use fuji::*;

let g = FujiAffine::gen_pallas();
let one = FujiField::one();

let r = msm::msm_eval(&[g, g], &[one, one], FujiCurve::Pallas).unwrap();
assert!(!r.is_identity());

// Batch MSM
let results = msm::msm_batch(&[1, 1], &[g, g], &[one, one], FujiCurve::Pallas).unwrap();
assert_eq!(results.len(), 2);
```
