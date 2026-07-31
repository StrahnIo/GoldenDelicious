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

## Benchmarks

**Single-core** (1 thread), 4× MSM on Pallas with distinct SRS bases and random scalars:

| k | SW-4x | PRL-srs-batch-4x | speedup |
|---|-------|------------------|---------|
| 4  | 19.6 ms  | **3.3 ms**   | 5.9× |
| 5  | 18.1 ms  | **3.8 ms**   | 4.7× |
| 6  | 20.1 ms  | **4.0 ms**   | 5.0× |
| 7  | 31.6 ms  | **5.2 ms**   | 6.0× |
| 8  | 44.4 ms  | **6.8 ms**   | 6.5× |
| 9  | 57.4 ms  | **11.4 ms**  | 5.1× |
| 10 | 110.4 ms | **14.2 ms**  | 7.8× |
| 11 | 223.3 ms | **26.7 ms**  | **8.4×** |
| 12 | 313.7 ms | **51.9 ms**  | 6.0× |
| 13 | 721.6 ms | **129.9 ms** | 5.6× |
| 14 | 819.4 ms | **200.3 ms** | 4.1× |

**Large k** — SW-4x on 10 threads vs PRL-srs-batch-4x on 1 thread:

| k | SW-4x (10 threads) | PRL-srs-batch-4x (1 thread) |
|---|--------------------|-----------------------------|
| 18 | 1943.968 ms | 2960.724 ms |
| 19 | 3576.443 ms | 5921.016 ms |
| 20 | 8723.267 ms | 11733.876 ms |

SW-4x = 4 sequential `best_multiexp` calls (pasta_curves NEON Montgomery, Rayon).
PRL-srs-batch-4x = single `prl_pippenger_batch_4` FFI call over 4 independent random scalar
sets sharing the same SRS bases.

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