# GoldenDelicious — Halo2 + Fuji PRL

A fork of [zcash/halo2](https://github.com/zcash/halo2) with PRL-accelerated MSM
via the Fuji crate.

**⚠️ In development.** MSM benchmarks work. Full proving/verification pipeline integration
is a work in progress.

## What this is

This is the **GoldenDelicious** upgrade for Halo2. It replaces the scalar-field
multiplication inside Halo2's MSM (multi-scalar multiplication) with a
4-way interleaved CIOS Montgomery PRL engine, achieving **up to 2.3× faster MSM**
than `pasta_curves` NEON Montgomery on Apple Silicon (M1–M4) for random scalars at k=11 (2048 points).

## Benchmarks

| Benchmark | k | SW-4x | SW-identg-4x | PRL-identg-4x | PRL-4x | PRL-batch-4x |
|-----------|----|-------|-------------|--------------|--------|-------------|
| 4× MSM | 8 | 7.4 ms | 7.7 ms | **8.2 ms** | 10.1 ms | 9.6 ms |
| 4× MSM | 11 | 31.6 ms | 33.7 ms | **15.1 ms** | 30.6 ms | 29.7 ms |
| 4× MSM | 12 | 59.8 ms | 62.2 ms | **30.1 ms** | 54.3 ms | 60.0 ms |
| Single MSM (k=11, via eval) | 11 | 4.66 ms\* | — | — | **2.12 ms** | — |

\* Software baseline via `best_multiexp` (pasta_curves NEON Montgomery). Fuji single-MSM via `MSM::eval()`.

**Key result:** PRL-identg-4x (identical G bases, random scalars) is **2.1× faster** than `best_multiexp` with identical G bases at k=11.

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