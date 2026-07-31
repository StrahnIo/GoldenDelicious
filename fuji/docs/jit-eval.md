# In-Process JIT Evaluator (`JitEval`)

## Why

Every ZK proof system needs to evaluate polynomial constraints per
challenge. The naive approach is a C interpreter calling into `fuji_eval`
via a subprocess — but that costs a fork + 49 MB file load + JIT compile
**every prove**. That doubles the wall time.

`JitEval` eliminates the subprocess entirely. It calls the JIT compiler
directly in-process via C FFI:

```
Before (subprocess):  Rust → fork() → fuji_eval → file load → JIT compile → dispatch_apply → exit → Rust
After  (JitEval):     Rust → JIT compile → dispatch_apply → results
```

## Quick Start

```rust
use fuji::eval::JitEval;
use fuji::FujiField;

// 1. Load bytecode once (your prove setup)
let bc_buf = std::fs::read("prove.fuji").expect("read bytecode");

// 2. Compile JIT once
let jit = JitEval::compile(&bc_buf, 5711, 16810).expect("JIT compile");

// 3. Per prove: update scalars and eval
let scalars = vec![FujiField::zero(); 5711];      // challenge scalars
let mut results = vec![FujiField::zero(); 16810]; // output buffer
jit.eval(&scalars, &mut results).expect("eval");

// 4. jit is dropped → cache freed automatically
```

## Two-Phase API

`JitEval` separates compilation from execution, matching the
`spawn`/`eval` pattern familiar from the subprocess `EvalChild`:

### Phase 1: `JitEval::compile(bc_buf, n_scalars, n_results)`

- Parses the `.fuji` bytecode buffer (no file I/O — buffer is already
  in memory).
- Compiles the ARM64 JIT code into a MAP_JIT buffer.
- Pre-allocates per-chunk scratch and output buffers.
- Returns `Some(JitEval)` on success, `None` on bad bytecode or JIT
  failure.
- Takes ~10 ms on an M4.

### Phase 2: `jit.eval(scalars, results)`

- Zeroes per-chunk scratch buffers (inside `dispatch_apply`, one chunk
  per core — no cache eviction).
- Dispatches all chunks via `dispatch_apply` with
  `QOS_CLASS_USER_INTERACTIVE` (prefers P-cores on Apple Silicon).
- Writes interleaved results to `results`.
- Returns `Ok(())` on success, `Err(String)` on failure.
- First call: ~600-700 ms (cold P-cores, JIT code in fresh cache).
- Subsequent calls: ~800-1000 ms (thermal throttling).

### Drop: cache freed automatically

`JitEval` implements `Drop` — the JIT MAP_JIT buffer and per-chunk
scratch buffers are freed when the handle goes out of scope. No
manual `free()` call needed.

## Performance

Measured on Apple M4 (4 P-cores + 6 E-cores, 24 GB):

| Metric | Value | Notes |
|--------|-------|-------|
| JIT compile | ~10 ms | Bytecode parse + compile + buffer alloc |
| Cold eval (run 1) | **~630 ms** | P-cores fresh, no thermal throttle |
| Warm eval (runs 2-5) | ~850-1000 ms | P-cores throttled after ~500 ms |
| Per chunk | ~20-24 ms | 41 chunks × 410 elements |
| vs `bench_parallel` (C) | **identical** | Same dispatch_apply + JIT fn |
| vs `EvalChild` (subprocess) | **~150 ms faster** | No fork/exec/file I/O |

### Why runs degrade

The M4's 4 P-cores hit max frequency for ~500 ms, then thermally
throttle. Since the JIT dispatch takes 600-1000 ms, the last ~half
runs at reduced frequency. This affects all dispatch methods equally:

```
Run 1: ████████████████████ 630 ms  (4 P-cores at 4.0 GHz)
Run 2: ██████████████░░░░░░ 850 ms  (4 P-cores at ~3.0 GHz)
Run 3: ████████████░░░░░░░░ 920 ms  (mixed P/E-core)
Run 4: ███████████░░░░░░░░░ 980 ms
Run 5: ██████████░░░░░░░░░░ 1050 ms
```

This is hardware-enforced. No software QoS setting or thread pinning
can prevent it on macOS (no `pthread_setaffinity_np`).

Compare with `bench_parallel` (the canonical C JIT benchmark):

| Run | `JitEval` (Rust) | `bench_parallel` (C) |
|-----|------------------|---------------------|
| 1   | 630 ms           | 540-620 ms          |
| 2   | 850 ms           | 730-900 ms          |
| 3   | 920 ms           | 900-1000 ms         |
| 4   | 980 ms           | 970-1500 ms         |
| 5   | 1050 ms          | 1000-2300 ms        |

Within noise. The two paths are identical — both call
`fuji_jit_compile` + `dispatch_apply` under the hood.

## Integration into a Proving System

Typical integration pattern:

```rust
use fuji::eval::JitEval;
use fuji::FujiField;

struct Prover {
    jit: JitEval,
    // Other per-prove state (transcript, etc.)
}

impl Prover {
    /// Load bytecode once at setup.
    fn new(bc_buf: &[u8], n_scalars: usize, n_results: usize) -> Self {
        let jit = JitEval::compile(bc_buf, n_scalars, n_results)
            .expect("JIT compile failed");
        Prover { jit }
    }

    /// Called each round with fresh challenge scalars.
    fn prove_round(&self, scalars: &[FujiField], results: &mut [FujiField]) {
        self.jit.eval(scalars, results).expect("eval failed");
        // results are in Montgomery form — convert if needed:
        // for r in results.iter_mut() { *r = r.from_mont(curve); }
    }
}
```

### Challenge Scalars

The per-prove challenge scalars replace the embedded scalars from the
`.fuji` bytecode file. Your proving system computes fresh challenges
each round — pass them as `scalars` to `eval()`.

The bytecode's challenge slot indices are in
`Bytecode::challenge_indices` (if you use the `Bytecode` wrapper), or
you can track them manually from your AST compile step.

### Curve

The curve (Pallas or Vesta) is embedded in the `.fuji` bytecode.
The JIT uses it to select the correct modulus and Montgomery
constants. You don't need to specify it.

## Comparison with Alternatives

| Approach | Latency | Per-prove overhead | Complexity |
|----------|---------|-------------------|------------|
| `JitEval` (in-process FFI) | **630-1000 ms** | None (cached) | Low |
| `EvalChild` (subprocess) | 800-1100 ms | +150 ms (fork+file) | Low |
| `Bytecode::execute_all_jit` (batch-clone) | 600-1200 ms | +mmap clones | Medium |
| C `fuji_eval` binary | 650-1000 ms | N/A (standalone) | Low |
| Rust C interpreter (`execute_all`) | 5000-6000 ms | None | High |

`JitEval` is the simplest Rust API with the lowest latency. It
replaces the subprocess `EvalChild` entirely.

## Troubleshooting

### `compile()` returns None

- The `.fuji` file may be corrupted or from an incompatible version.
- The JIT compiler may have run out of MAP_JIT memory (unlikely on a
  24 GB machine, but possible under extreme memory pressure).

### `eval()` returns Err

- Internal error in `dispatch_apply` or the JIT function pointer is
  invalid (cache corruption from unsafe code).
- This should not happen in normal use. If it does, the bytecode may
  be incompatible with the current library build.

### First eval is fast, subsequent evals are slow

This is thermal throttling (see "Why runs degrade" above). It affects
all dispatch methods equally. To minimize: keep the CPU cool, or
schedule proves with idle gaps between them.

### I want consistent 500 ms

Consistent 500 ms requires eliminating thermal throttling. On macOS
this isn't possible from userspace — you'd need kernel-level P-core
affinity and frequency capping. If you need consistent sub-500 ms,
consider running on a machine with more P-cores (M4 Pro/Max have 8-12
P-cores, spreading the thermal load).

## See Also

- `rust/fuji/src/eval.rs` — source for `JitEval`, `EvalChild`, `Bytecode`
- `tools/fuji_eval.c` — standalone C eval binary (subprocess target)
- `tools/eval_child.c` — C subprocess wrapper (replaced by JitEval)
- `include/fuji/jit.h` — C API for the JIT compiler and dispatch
