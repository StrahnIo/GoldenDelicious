# Build

## Prerequisites

- Apple Silicon Mac (M1–M4)
- macOS 14.0+
- Rust 1.75+ (stable)
- Xcode Command Line Tools (`xcode-select --install`)
- `clang` with arm64 support

## 1. Build the C Library

```shell
git clone https://github.com/anomalyco/fuji
cd fuji
make                          # produces libfuji.dylib
make test_field_ref           # optional: run C tests
```

This produces `libfuji.dylib` in the project root. The Rust build script
(`fuji-sys/build.rs`) searches parent directories from `CARGO_MANIFEST_DIR`
to find it.

## 2. Build the Rust Crates

```shell
cd rust
cargo build                    # builds fuji-sys + fuji + fuji-pasta
```

### Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `FUJI_LIB_DIR` | Path to directory containing `libfuji.dylib` | `../../` from `fuji-sys/` |

## 3. Run Tests

```shell
cd rust

# All tests
DYLD_LIBRARY_PATH=../.. cargo test -- --test-threads=1

# Just the trait bridge tests
DYLD_LIBRARY_PATH=../.. cargo test -p fuji-pasta -- --test-threads=1

# Just the core FFI tests
DYLD_LIBRARY_PATH=../.. cargo test -p fuji -- --test-threads=1
```

### Why `--test-threads=1`?

The AMX coprocessor and SME ZA tile registers are shared per CPU cluster
(not per core). When multiple threads issue AMX or SME instructions
concurrently, their register state collides and causes crashes.
Single-threaded execution avoids this.

## 4. Build Documentation

```shell
cd rust
cargo doc --no-deps --open     # opens in browser (crate-level docs)
# or open docs/ directory       # markdown docs (this directory)
```

## Workspace Layout

```
rust/
├── Cargo.toml                    # Workspace root
├── fuji-sys/                     # Raw FFI (no_std)
├── fuji/                         # Safe wrappers
├── fuji-pasta/                   # ff + group trait impls
└── docs/                         # This documentation
```

## Common Errors

### `dyld: Library not loaded: @rpath/libfuji.dylib`

The dynamic linker can't find `libfuji.dylib` at runtime. Either:

```shell
# Option A: set DYLD_LIBRARY_PATH
DYLD_LIBRARY_PATH=../.. cargo run

# Option B: copy next to the binary
cp ../libfuji.dylib target/debug/
```

### AMX / SME not available

On Intel Macs, VMs (Parallels, UTM), or Rosetta 2 the AMX coprocessor is
not exposed. `fuji_amx_available()` returns false and `fuji_field_mul` aborts
with a diagnostic message. AMX is **required** — there is no scalar fallback.

SME is only available on Apple M4 and later. Use `fuji_sme_available()`
to detect it at runtime.

### `fuji-pasta` compilation errors

If you see errors about `pasta_curves` types or FFI imports, ensure the
C library is built (`make` in the project root) and `DYLD_LIBRARY_PATH`
is set correctly.
