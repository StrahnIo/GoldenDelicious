# GoldenDelicious — Halo2 + Fuji PRL

A fork of [zcash/halo2](https://github.com/zcash/halo2) with PRL-accelerated MSM
via the Fuji crate.

**⚠️ In development.** MSM benchmarks work. Full proving/verification pipeline integration
is a work in progress.

## What this is

This is the **GoldenDelicious** upgrade for Halo2. It replaces the scalar-field
multiplication inside Halo2's MSM (multi-scalar multiplication) with a
4-way interleaved CIOS Montgomery PRL engine, achieving **up to 2.36× faster MSM**
than the state of the art on 4 P-cores, with **2.8× better performance-per-watt**,
on Apple Silicon (M1–M4).

## Performance

Batch MSM throughput (`fuji_prl_pippenger_batch_4`, 4 MSMs, distinct SRS generators,
Apple M4):

### Latency

| k | n | SoTA 4c | SoTA 10c (4P+6E) | FUJI 4c | vs SoTA 4c | vs SoTA 10c |
|---|----|---------|------------------|---------|------------|-------------|
| 11 | 2048 | 39 ms | 25 ms | **16.5 ms** | **2.36×** | **1.52×** |
| 12 | 4096 | 69 ms | 44 ms | **29.4 ms** | **2.35×** | **1.50×** |
| 13 | 8192 | 127 ms | 71 ms | **58 ms** | **2.19×** | **1.22×** |
| 14 | 16384 | 232 ms | 126 ms | **113 ms** | **2.05×** | **1.12×** |

### Efficiency

| Metric | SoTA 4c | SoTA 10c | FUJI 4c |
|--------|---------|----------|---------|
| Cores | 4P | 4P+6E | 4P |
| Real time | 2.29 s | 2.10 s | 2.48 s |
| Total energy | 11.77 J | 24.97 J | 14.14 J |
| Avg power | 5.14 W | 11.89 W | 5.70 W |
| J/B instruction | 0.59 | 1.24 | **0.40** |
| IPC | 2.57 | 2.31 | **2.98** |
| Peak memory | 12.2 MB | 17.2 MB | 22.2 MB |

### Energy-delay product (k=14)

| Metric | SoTA 4c | SoTA 10c | FUJI 4c |
|--------|---------|----------|---------|
| Task time | 232 ms | 126 ms | **113 ms** |
| Energy per task | 1.19 J | 1.50 J | **0.64 J** |
| EDP (J·s) | 0.2766 | 0.1888 | **0.0728** |
| **Perf-per-watt score** | **100** | **147** | **380** |

## Setup

See [GUIDE_DEV.md](GUIDE_DEV.md) for full development setup instructions.

Quick start:

```bash
# Link the Fuji crate from https://github.com/StrahnIo/FUJI
ln -s /path/to/fuji_repo/rust fuji

# Place the precompiled binary (downloaded from releases)
mkdir libfuji && cp /path/to/libfuji_c.a libfuji/

# Build and bench
export FUJI_LIB_DIR=$PWD/libfuji
RAYON_NUM_THREADS=4 rustup run stable cargo bench --features fuji --bench msm_4x
```

## Status

| Feature | Status |
|---------|--------|
| MSM (`prl_pippenger`) | ✅ Working |
| Batch MSM (`prl_msm_batch`) | ✅ Working |
| Pre-built MSM vectors in `MSM::eval()` | ✅ Working |
| SRS precompute context | 🔧 Infrastructure in place |
| NTT (`fuji_ntt`) | 🔧 Safe wrapper, dispatch disabled |
| Lockstep batch (`fuji_srs_commit_batch`) | 🔧 Needs correction |
| Polynomial commit batching | 🔧 Not yet wired |

## Minimum Supported Rust Version

Requires Rust **1.60** or higher.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.