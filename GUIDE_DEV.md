# Halo2 Fuji Development Guide

Setup guide for developing with the Fuji PRL-accelerated MSM backend on Apple Silicon.

## Prerequisites

- Apple Silicon Mac (M1–M4)
- macOS 14.0+
- Rust toolchain (stable)
- Xcode Command Line Tools

## Repository Structure

```
halo2/
├── fuji/                              # Symlink → fuji_crate/rust/
│   ├── fuji/                          # Safe Rust wrappers
│   ├── fuji-sys/                      # Raw FFI bindings
│   └── fuji-pasta/                    # ff/group trait bridge
├── libfuji/                           # Precompiled C library
│   ├── libfuji_c.a                    # Fuji AMX/PRL binary
│   └── libfuji.a                      # Symlink → libfuji_c.a
├── halo2_proofs/
│   ├── benches/
│   │   ├── msm.rs                     # Single MSM benchmarks
│   │   ├── msm_4x.rs                  # 4× MSM batch benchmarks
│   │   └── plonk.rs                   # Full proof pipeline benchmarks
│   └── src/
│       ├── arithmetic/
│       │   └── fuji.rs                # Fuji dispatch (MSM + NTT)
│       └── poly/
│           └── commitment/
│               └── msm.rs             # Pre-built Fuji MSM vectors
└── Cargo.toml
```

## Setup

### 1. Clone Repositories

```bash
git clone <fuji-crate-url> fuji_crate     # Apple's fuji repo
git clone <halo2-fork-url> halo2           # This repo
```

The fuji crate lives in `fuji_crate/rust/` (contains `fuji/`, `fuji-sys/`, `fuji-pasta/`).

### 2. Link the Fuji Crate

Create a symlink so the workspace can find the fuji source:

```bash
cd halo2
ln -s /absolute/path/to/fuji_crate/rust fuji
```

### 3. Place the Precompiled Binary

The proprietary AMX/PRL C code is distributed as a static library:

```bash
mkdir libfuji
cp /path/to/libfuji_c.a libfuji/
ln -s libfuji_c.a libfuji/libfuji.a
```

### 4. Verify the Link

```bash
nm libfuji/libfuji_c.a | grep fuji_f_mul_4x
# Should show: 0000000000000234 T _fuji_f_mul_4x
```

## Building

```bash
export FUJI_LIB_DIR=/absolute/path/to/halo2/libfuji

# Build with Fuji:
cargo build --features fuji -p halo2_proofs

# Build without Fuji (software-only):
cargo build -p halo2_proofs
```

## Running Benchmarks

### Single MSM Benchmarks

```bash
export FUJI_LIB_DIR=/path/to/halo2/libfuji

# Full comparison (SW + Fuji):
cargo bench --features fuji --bench msm

# Software-only baseline:
cargo bench --bench msm

# Quick test with 4 threads (P-cores only):
RAYON_NUM_THREADS=4 cargo bench --features fuji --bench msm
```

Output groups (all in one `msm/` group):

| Benchmark | Scalars | Bases | Notes |
|-----------|---------|-------|-------|
| `msm/sw/11` | Random Fq | Distinct SRS | Software baseline |
| `msm/sw-all1/11` | All-1 Fq | Distinct SRS | Best-case SW |
| `msm/fuji-prl/11` | Random Fq | Distinct SRS | Full integration path |
| `msm/fuji-all1/11` | All-1 FujiField | Distinct SRS Mont | Direct `prl_pippenger` |
| `msm/fuji-apple-identg/11` | Random Fq | Identical G (Mont) | Apple-style benchmark |

### 4× MSM Benchmarks

```bash
export FUJI_LIB_DIR=/path/to/halo2/libfuji
cargo bench --features fuji --bench msm_4x
```

Output (plain timing, no criterion):

| Benchmark | Description |
|-----------|-------------|
| `sw-4x/k=12` | 4× sequential `best_multiexp` (distinct SRS) |
| `sw-identg-4x/k=12` | 4× sequential `best_multiexp` (identical G) |
| `prl-identg-4x/k=12` | 4× sequential `prl_pippenger` (identical G, Mont) |
| `prl-4x/k=12` | 4× sequential `prl_pippenger` (distinct SRS, Mont) |
| `prl-batch-4x/k=12` | 1× `prl_msm_batch` (4 MSMs, Mont) |

### Full Proof Pipeline (PLONK)

```bash
export FUJI_LIB_DIR=/path/to/halo2/libfuji

# With Fuji:
cargo bench --features fuji --bench plonk

# Software-only:
cargo bench --bench plonk
```

Output groups:

| Benchmark | Description |
|-----------|-------------|
| `plonk-keygen/k=8` | Key generation (same with/without Fuji) |
| `plonk-prover-sw/k=8` | Proving time (software path) |
| `plonk-verifier-sw/k=8` | Verification time (software path) |
| `plonk-prover-fuji/k=8` | Proving time (Fuji-accelerated) |
| `plonk-verifier-fuji/k=8` | Verification time (Fuji-accelerated) |

## Architecture

### Integration Points

| Call Site | File | What it does |
|-----------|------|--------------|
| `MSM::eval()` | `msm.rs` | Verifier: checks accumulated MSM via `try_multiexp_precomputed` |
| `Params::commit()` | `commitment.rs` | Prover: commits polynomial via `try_multiexp` |
| `Params::commit_lagrange()` | `commitment.rs` | Prover: commits Lagrange polynomial via `try_multiexp` |
| `Params::new()` | `commitment.rs` | Init: precomputes SRS context in Montgomery form |

### Data Flow

```
commit(poly, blind)
  → field_to_fuji(coeffs)               // Normal-form scalars
  → bases[i].to_mont(curve)              // Mont-form SRS generators
  → fuji::msm::prl_pippenger(scalars, bases_mont, curve)
  → result.from_mont(curve)              // Back to normal form
  → C::Curve (pasta_curves type)
```

### Key Parameters

| Parameter | Value |
|-----------|-------|
| Window size | 8 bits (256 buckets) |
| Windows | 32 (255 bits) |
| Min MSM for Fuji | 256 scalars |
| Basis form | Montgomery (to_mont at entry, from_mont at exit) |
| Scalar form | Normal (byte-level window extraction) |

## Troubleshooting

### `dyld: Library not loaded`

The linker can't find `libfuji.dylib`. Either:
```bash
# Option A: set DYLD_LIBRARY_PATH
export DYLD_LIBRARY_PATH=/path/to/halo2/libfuji

# Option B: use static linking (default — just need FUJI_LIB_DIR)
export FUJI_LIB_DIR=/path/to/halo2/libfuji
```

### `symbol(s) not found for architecture arm64`

The `.a` file is missing the required PRL symbols. Verify:
```bash
nm /path/to/libfuji_c.a | grep fuji_f_mul_4x
nm /path/to/libfuji_c.a | grep fuji_prl_available
```

### Rayon performance

The `prl_pippenger` uses `par_chunks_mut` over 32 windows. At k ≤ 12, serial fill is faster:
```bash
RAYON_NUM_THREADS=1 cargo bench --features fuji --bench msm
```

### AMX vs PRL

The C library uses **PRL** (parallel CIOS Montgomery) for field multiplication, not AMX.
AMX was found to be ~545× slower than NEON Montgomery for 255-bit field mul.
PRL uses 4× interleaved CIOS via `fuji_f_mul_4x` at ~17ns per mul.
