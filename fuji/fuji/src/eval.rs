//! Flat-instruction evaluator for Halo2 AST synthesis.
//!
//! Compile the evaluator's recursive AST into a linear `Vec<Instr>` once per
//! prove, then execute it per chunk. This eliminates recursive dispatch
//! overhead (~50% of evaluator time).
//!
//! # Usage (for any Halo2 backend)
//!
//! 1. **Compile** the AST once per prove:
//! ```ignore
//! fn compile(ast: &Ast) -> Vec<Instr> { /* walk AST, emit Instr */ }
//! ```
//!
//! 2. **Prepare context** once per prove:
//! ```ignore
//! let ctx = EvalContext {
//!     chunk_len: 410,
//!     chunk_start: 0,            // updated per chunk
//!     max_depth: estimate_depth(&code),
//!     poly_table: vec![PolyEntry { data, domain_size }],
//!     omega: domain.get_omega(),
//!     scalars: vec![],           // filled by compile()
//! };
//! ```
//!
//! 3. **Execute** per chunk:
//! ```ignore
//! let mut scratch = vec![vec![FujiField::zero(); ctx.chunk_len]; ctx.max_depth];
//! let mut out = vec![FujiField::zero(); ctx.chunk_len];
//! for chunk in 0..num_chunks {
//!     ctx.chunk_start = chunk * ctx.chunk_len;
//!     execute_chunk(&code, &ctx, &mut scratch, &mut out);
//! }
//! ```
//!
//! # All operands in Montgomery form
//!
//! All polynomial data and scalar arguments must be in Montgomery form.
//! Results are returned in Montgomery form. Convert once at the batch boundary
//! using `.from_mont(curve)`.

use crate::FujiField;
use crate::FujiCurve;
use std::mem::size_of;
use std::time::Instant;

// libdispatch FFI (stable dispatch_apply_f, no deadlocks vs fuji_jit_dispatch)
extern "C" {
    fn dispatch_get_global_queue(priority: i64, flags: u64) -> *const std::ffi::c_void;
    fn dispatch_apply_f(
        iterations: usize,
        queue: *const std::ffi::c_void,
        context: *mut std::ffi::c_void,
        work: extern "C" fn(context: *mut std::ffi::c_void, idx: usize),
    );
}


struct DispatchCtx {
    fn_ptr: fuji_sys::jit_chunk_fn,
    clones: *const fuji_sys::jit_chunk_fn,
    n_clones: usize,
    scs: *mut *mut fuji_sys::fuji_field,
    outs: *mut *mut fuji_sys::fuji_field,
    poly_table: *const fuji_sys::fuji_eval_poly,
    scalars: *const fuji_sys::fuji_field,
    omega: *const fuji_sys::fuji_field,
    p: *const std::ffi::c_void,
    pinv: u64,
    chunk_len: usize,
}



const CURVE: FujiCurve = FujiCurve::Pallas;



/// A compiled flat instruction for the evaluator.
#[derive(Clone, Debug)]
pub enum Instr {
    /// Load a rotated polynomial column into scratch[sp].
    /// `col` indexes into `ctx.poly_table`. The slice is rotated by `rotation`
    /// rows (domain rows, multiplied by stride internally).
    LoadRotated { col: u32, rotation: i32 },

    /// scratch[sp-2] += scratch[sp-1]; sp -= 1
    Add,
    /// scratch[sp-2] -= scratch[sp-1]; sp -= 1
    Sub,
    /// scratch[sp-2] *= scratch[sp-1]; sp -= 1
    Mul,
    /// scratch[sp-1] *= scalar[scalar_idx]; scalar_idx += 1
    Scale { scalar_idx: u32 },
    /// out = out * base + term, where base = scratch[sp-1],
    /// term = scratch[sp-2], result → scratch[sp-2]; sp -= 1
    ///
    /// When `acc_row` ≠ 0xFFFFFFFF, the accumulator lives at that fixed
    /// scratch row instead of `sp-2`.  Term evaluations start above it
    /// and never reach `acc_row`, so the accumulator stays un-corrupted.
    Fma { scalar_idx: u32, acc_row: u32 },
    /// scratch[sp] = ω^(chunk_start + i) × scalar[scalar_idx]; sp += 1
    LinearTerm { scalar_idx: u32 },
    /// scratch[sp] = fill(scalar[scalar_idx]); sp += 1
    Constant { scalar_idx: u32 },
    /// scratch[sp-1] = -scratch[sp-1]
    Negate,
    /// out = scratch[src_row]; 0xFFFFFFFF means scratch[0]
    CopyToOut { src_row: u32 },
    /// scratch[sp] = scratch[sp-1].clone(); sp += 1
    Dup,
}

/// Per-prove context shared across all chunks.
#[derive(Clone)]
pub struct EvalContext {
    /// Element count in this chunk.
    pub chunk_len: usize,
    /// Starting element offset for the current chunk (updated per chunk).
    pub chunk_start: usize,
    /// Estimated or computed maximum scratch depth.
    pub max_depth: usize,
    /// Polynomial column table: data in Mont form, contiguous per column.
    pub poly_table: Vec<PolyEntry>,
    /// Extended-domain primitive root of unity (Mont form).
    pub omega: FujiField,
    /// Flat array of scalar arguments. Indexed by `Instr::Scale`, `Fma`,
    /// `LinearTerm`, `Constant` via their `scalar_idx` field.
    pub scalars: Vec<FujiField>,
}

/// Entry for one polynomial column.
#[derive(Clone)]
pub struct PolyEntry {
    /// Polynomial data in Montgomery form, length = domain_size.
    pub data: Vec<FujiField>,
    /// Domain size (number of elements in the full polynomial).
    pub domain_size: usize,
}

/// Pre-converted C data for the flat evaluator — cache this across chunks.
pub struct CachedEval {
    pub c_code: Vec<fuji_sys::fuji_eval_instr>,
    pub c_polys: Vec<fuji_sys::fuji_eval_poly>,
}

/// Build cached C data from a compiled instruction sequence and context.
/// Call once before the chunk loop (not per chunk).
pub fn build_cached(code: &[Instr], ctx: &EvalContext) -> CachedEval {
    let c_code = code.iter().map(|instr| {
        let (op, a0, a1) = match instr {
            Instr::LoadRotated { col, rotation } => (0, *col as i32, *rotation),
            Instr::Add => (1, 0, 0),
            Instr::Sub => (2, 0, 0),
            Instr::Mul => (3, 0, 0),
            Instr::Scale { scalar_idx } => (4, *scalar_idx as i32, 0),
            Instr::Fma { scalar_idx, acc_row } => (5, *scalar_idx as i32, *acc_row as i32),
            Instr::LinearTerm { scalar_idx } => (6, *scalar_idx as i32, 0),
            Instr::Constant { scalar_idx } => (7, *scalar_idx as i32, 0),
            Instr::Negate => (8, 0, 0),
            Instr::CopyToOut { src_row } => (9, *src_row as i32, 0),
            Instr::Dup => (10, 0, 0),
        };
        fuji_sys::fuji_eval_instr { opcode: op, arg0: a0, arg1: a1 }
    }).collect();

    let c_polys = ctx.poly_table.iter().map(|p| {
        fuji_sys::fuji_eval_poly {
            data: p.data.as_ptr() as *const fuji_sys::fuji_field,
            domain_size: p.domain_size as i32,
        }
    }).collect();

    CachedEval { c_code, c_polys }
}

/// Execute one chunk using cached C data and a flat scratch buffer.
///
/// `flat_scratch` must have at least `max_depth * chunk_len` elements.
/// `out` must have at least `chunk_len` elements. Results in Montgomery form.
/// No per-chunk heap allocations.
pub fn execute_cached(
    cache: &CachedEval,
    ctx: &EvalContext,
    flat_scratch: &mut [FujiField],
    out: &mut [FujiField],
) {
    let n_rows = ctx.max_depth as i32;
    let len = ctx.chunk_len as i32;
    unsafe {
        fuji_sys::fuji_eval_execute(
            cache.c_code.as_ptr(), cache.c_code.len() as i32,
            cache.c_polys.as_ptr(), cache.c_polys.len() as i32,
            ctx.scalars.as_ptr() as *const fuji_sys::fuji_field,
            ctx.scalars.len() as i32,
            flat_scratch.as_mut_ptr() as *mut fuji_sys::fuji_field, n_rows,
            ctx.chunk_start as i32, len,
            &ctx.omega as *const FujiField as *const fuji_sys::fuji_field,
            CURVE as i32,
            out.as_mut_ptr() as *mut fuji_sys::fuji_field,
            std::ptr::null_mut(),
        );
    }
}

/// Execute all chunks in parallel using rayon.
///
/// Builds cached C data once, warms the C chain cache with chunk 0,
/// then dispatches remaining chunks across rayon threads.
///
/// `n` = extended domain size. Output buffer must have `n` elements.
/// Results in Montgomery form. Convert once at the batch boundary.
pub fn execute_all(
    code: &[Instr],
    ctx: &EvalContext,
    poly_table: &[PolyEntry],
    scalars: &[FujiField],
    n: usize,
    results: &mut [FujiField],
) {
    let chunk_len = ctx.chunk_len;
    let max_depth = ctx.max_depth;
    let num_chunks = (n + chunk_len - 1) / chunk_len;
    let scratch_size = max_depth * chunk_len;

    // Build C cache once (shared by all chunks)
    let c_code: Vec<fuji_sys::fuji_eval_instr> = code.iter().map(|instr| {
        let (op, a0, a1) = match instr {
            Instr::LoadRotated { col, rotation } => (0, *col as i32, *rotation),
            Instr::Add => (1, 0, 0),
            Instr::Sub => (2, 0, 0),
            Instr::Mul => (3, 0, 0),
            Instr::Scale { scalar_idx } => (4, *scalar_idx as i32, 0),
            Instr::Fma { scalar_idx, acc_row } => (5, *scalar_idx as i32, *acc_row as i32),
            Instr::LinearTerm { scalar_idx } => (6, *scalar_idx as i32, 0),
            Instr::Constant { scalar_idx } => (7, *scalar_idx as i32, 0),
            Instr::Negate => (8, 0, 0),
            Instr::CopyToOut { src_row } => (9, *src_row as i32, 0),
            Instr::Dup => (10, 0, 0),
        };
        fuji_sys::fuji_eval_instr { opcode: op, arg0: a0, arg1: a1 }
    }).collect();

    let c_polys: Vec<fuji_sys::fuji_eval_poly> = poly_table.iter().map(|p| {
        fuji_sys::fuji_eval_poly {
            data: p.data.as_ptr() as *const fuji_sys::fuji_field,
            domain_size: p.domain_size as i32,
        }
    }).collect();

    let cache = CachedEval { c_code, c_polys };
    let scalars_ptr_u = scalars.as_ptr() as usize;
    let n_scalars = scalars.len() as i32;
    let n_rows = max_depth as i32;
    let curve: i32 = CURVE as i32;
    let omega_ptr_u = &ctx.omega as *const FujiField as *const fuji_sys::fuji_field as usize;

    // Warmup: call fuji_eval_execute once to populate the C chain cache.
    // This also processes chunk 0, so we write its results.
    {
        let mut warmup = vec![FujiField::zero(); scratch_size];
        let out0 = &mut results[..chunk_len];
        let code_ptr = cache.c_code.as_ptr();
        let code_len = cache.c_code.len() as i32;
        let polys_ptr = cache.c_polys.as_ptr();
        let polys_len = cache.c_polys.len() as i32;
        unsafe {
            fuji_sys::fuji_eval_execute(
                code_ptr, code_len, polys_ptr, polys_len,
                scalars_ptr_u as *const fuji_sys::fuji_field, n_scalars,
                warmup.as_mut_ptr() as *mut fuji_sys::fuji_field, n_rows,
                0, chunk_len as i32,
                omega_ptr_u as *const fuji_sys::fuji_field, curve,
                out0.as_mut_ptr() as *mut fuji_sys::fuji_field,
                std::ptr::null_mut(),
            );
        }
    }

    // Cast pointers to usize (which is Send) for the closure.
    let code_ptr_u = cache.c_code.as_ptr() as usize;
    let code_len = cache.c_code.len() as i32;
    let polys_ptr_u = cache.c_polys.as_ptr() as usize;
    let polys_len = cache.c_polys.len() as i32;
    let scalars_ptr_u = scalars.as_ptr() as *const fuji_sys::fuji_field as usize;
    let omega_ptr_u = &ctx.omega as *const FujiField as *const fuji_sys::fuji_field as usize;

    let results_ptr = results.as_mut_ptr() as usize;
    if num_chunks > 1 {
        rayon::scope(|s| {
            for chunk in 1..num_chunks {
                s.spawn(move |_| {
                    let mut scratch = vec![FujiField::zero(); scratch_size];
                    let chunk_start = (chunk * chunk_len) as i32;
                    let out_ptr = (results_ptr + chunk * chunk_len * size_of::<FujiField>())
                        as *mut fuji_sys::fuji_field;
                    unsafe {
                        fuji_sys::fuji_eval_execute(
                            code_ptr_u as *const fuji_sys::fuji_eval_instr, code_len,
                            polys_ptr_u as *const fuji_sys::fuji_eval_poly, polys_len,
                            scalars_ptr_u as *const fuji_sys::fuji_field, n_scalars,
                            scratch.as_mut_ptr() as *mut fuji_sys::fuji_field, n_rows,
                            chunk_start, chunk_len as i32,
                            omega_ptr_u as *const fuji_sys::fuji_field, curve,
                            out_ptr,
                            std::ptr::null_mut(),
                        );
                    }
                });
            }
        });
    }
}

/// JIT-compiled evaluator — compile once, call per chunk.
pub struct CachedJit {
    pub fn_ptr: fuji_sys::jit_chunk_fn,
    /// Cached poly table — must stay alive for JIT function calls.
    pub c_polys: Vec<fuji_sys::fuji_eval_poly>,
    /// Scratch depth (from estimate_depth or as passed to build_jit).
    pub max_depth: usize,
}

/// Build a JIT-compiled evaluator from bytecode.
/// Call once before the chunk loop. The returned `CachedJit` must be freed via
/// `CachedJit::free()` after all chunks are processed.
pub fn build_jit(
    code: &[Instr],
    poly_table: &[PolyEntry],
    chunk_len: usize,
    max_depth: usize,
) -> CachedJit {
    let c_code: Vec<fuji_sys::fuji_eval_instr> = code.iter().map(|instr| {
        let (op, a0, a1) = match instr {
            Instr::LoadRotated { col, rotation } => (0, *col as i32, *rotation),
            Instr::Add => (1, 0, 0),
            Instr::Sub => (2, 0, 0),
            Instr::Mul => (3, 0, 0),
            Instr::Scale { scalar_idx } => (4, *scalar_idx as i32, 0),
            Instr::Fma { scalar_idx, acc_row } => (5, *scalar_idx as i32, *acc_row as i32),
            Instr::LinearTerm { scalar_idx } => (6, *scalar_idx as i32, 0),
            Instr::Constant { scalar_idx } => (7, *scalar_idx as i32, 0),
            Instr::Negate => (8, 0, 0),
            Instr::CopyToOut { src_row } => (9, *src_row as i32, 0),
            Instr::Dup => (10, 0, 0),
        };
        fuji_sys::fuji_eval_instr { opcode: op, arg0: a0, arg1: a1 }
    }).collect();

    let c_polys: Vec<fuji_sys::fuji_eval_poly> = poly_table.iter().map(|p| {
        fuji_sys::fuji_eval_poly {
            data: p.data.as_ptr() as *const fuji_sys::fuji_field,
            domain_size: p.domain_size as i32,
        }
    }).collect();

    let fn_ptr = unsafe {
        fuji_sys::fuji_jit_compile(
            c_code.as_ptr(), c_code.len() as i32,
            c_polys.as_ptr(), c_polys.len() as i32,
            chunk_len as i32, max_depth as i32,
            CURVE as i32,
        )
    };

    // c_code can be dropped after compile (JIT doesn't reference it at runtime).
    drop(c_code);

    CachedJit { fn_ptr, c_polys, max_depth }
}

impl CachedJit {
    /// Free the JIT-compiled buffer. Call after all chunks are done.
    pub fn free(self) {
        unsafe { fuji_sys::fuji_jit_free(self.fn_ptr); }
    }

    /// Dispatch function (for use from halo2_proofs where fuji_sys isn't available).
    /// Calls the JIT function pointer per chunk with the given parameters.
    pub fn dispatch(
        &self,
        scalars: &[FujiField],
        omega: &FujiField,
        modulus: *const std::ffi::c_void,
        pinv: u64,
        chunk_len: usize,
        n: usize,
        results: &mut [FujiField],
    ) {
        let num_chunks = (n + chunk_len - 1) / chunk_len;
        let scratch_size = self.max_depth * (n / num_chunks);
        for ch in 0..num_chunks {
            let mut scratch = vec![FujiField::zero(); scratch_size];
            unsafe {
                (self.fn_ptr)(
                    (ch * chunk_len) as i32,
                    scratch.as_mut_ptr() as *mut fuji_sys::fuji_field,
                    results.as_mut_ptr().add(ch * chunk_len) as *mut fuji_sys::fuji_field,
                    self.c_polys.as_ptr(),
                    scalars.as_ptr() as *const fuji_sys::fuji_field,
                    omega as *const FujiField as *const fuji_sys::fuji_field,
                    modulus,
                    pinv,
                );
            }
        }
    }
}

/// Execute all chunks in parallel using the JIT compiler.
///
/// Compiles bytecode once via `fuji_jit_compile`, then batch-clones the JIT
/// code into per-thread MAP_JIT buffers under a single write-protect toggle
/// (avoiding the IC IVAU broadcast storm that destroys parallelism).
/// Dispatches all chunks across rayon threads, each thread executing from
/// its own clone to keep L1I lines independent.
///
/// `n` = extended domain size. Output buffer must have `n` elements.
/// Results in Montgomery form.
pub fn execute_all_jit(
    code: &[Instr],
    poly_table: &[PolyEntry],
    scalars: &[FujiField],
    omega: &FujiField,
    chunk_len: usize,
    max_depth: usize,
    n: usize,
    results: &mut [FujiField],
) {
    let num_chunks = (n + chunk_len - 1) / chunk_len;
    let scratch_size = max_depth * chunk_len;

    // Compile once
    let jit = build_jit(code, poly_table, chunk_len, max_depth);

    // ── Batch-clone: one toggle, many copies ──
    let code_size = unsafe { fuji_sys::fuji_jit_code_size() };
    let n_clones = std::cmp::min(rayon::current_num_threads(), num_chunks);

    // mmap clone buffers for threads 1..n_clones (thread 0 uses the original)
    let mut clones: Vec<fuji_sys::jit_chunk_fn> = Vec::with_capacity(n_clones);
    let mut clone_bufs: Vec<*mut libc::c_void> = Vec::with_capacity(n_clones);
    clones.push(jit.fn_ptr);
    clone_bufs.push(std::ptr::null_mut()); // index 0: no separate buffer (uses original)
    for _ in 1..n_clones {
        let buf = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                code_size,
                libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_JIT,
                -1,
                0,
            )
        };
        if buf == libc::MAP_FAILED {
            panic!("mmap MAP_JIT failed for JIT clone");
        }
        clone_bufs.push(buf);
        clones.push(unsafe { std::mem::transmute(buf) });
    }

    // Single write-protect toggle for all clones
    if n_clones > 1 {
        let rc = unsafe {
            fuji_sys::fuji_jit_batch_clone(
                jit.fn_ptr,
                code_size,
                clones[1..].as_mut_ptr(),
                (n_clones - 1) as i32,
            )
        };
        if rc != 0 {
            panic!("fuji_jit_batch_clone failed with code {}", rc);
        }
    }

    let p = unsafe { fuji_sys::fuji_field_modulus(CURVE as i32) as *const std::ffi::c_void };
    let p_inv_lo: u64 = match CURVE {
        FujiCurve::Pallas => 0x992d30ecffffffff,
        FujiCurve::Vesta => 0x8c46eb20ffffffff,
    };

    let polys_ptr = jit.c_polys.as_ptr();
    let scalars_ptr = scalars.as_ptr() as *const fuji_sys::fuji_field;
    let omega_ptr = omega as *const FujiField as *const fuji_sys::fuji_field;

    let results_ptr = results.as_mut_ptr() as usize;
    let polys_ptr_u = polys_ptr as usize;
    let scalars_ptr_u = scalars_ptr as usize;
    let omega_ptr_u = omega_ptr as usize;
    let p_u = p as usize;

    rayon::scope(|s| {
        for chunk in 0..num_chunks {
            let fn_ptr = clones[chunk % n_clones];
            s.spawn(move |_| {
                let mut scratch = vec![FujiField::zero(); scratch_size];
                let chunk_start = (chunk * chunk_len) as i32;
                let out_ptr = (results_ptr + chunk * chunk_len * size_of::<FujiField>())
                    as *mut fuji_sys::fuji_field;
                unsafe {
                    (fn_ptr)(
                        chunk_start,
                        scratch.as_mut_ptr() as *mut fuji_sys::fuji_field,
                        out_ptr,
                        polys_ptr_u as *const fuji_sys::fuji_eval_poly,
                        scalars_ptr_u as *const fuji_sys::fuji_field,
                        omega_ptr_u as *const fuji_sys::fuji_field,
                        p_u as *const std::ffi::c_void,
                        p_inv_lo,
                    );
                }
            });
        }
    });

    // Free clone buffers using original mmap addresses (index 0 is the original, freed below)
    for &buf in &clone_bufs[1..] {
        if !buf.is_null() {
            unsafe { libc::munmap(buf, code_size); }
        }
    }

    jit.free();
}

/// Legacy: execute one chunk with Vec<Vec> scratch (allocates per chunk).
pub fn execute_chunk(
    code: &[Instr],
    ctx: &EvalContext,
    scratch: &mut [Vec<FujiField>],
    out: &mut [FujiField],
) {
    let len = ctx.chunk_len;
    for row in scratch.iter_mut() { row.resize(len, FujiField::zero()); }
    let cache = build_cached(code, ctx);
    let mut flat_scratch: Vec<FujiField> = scratch.iter()
        .flat_map(|r| r.iter().copied())
        .collect();
    unsafe {
        fuji_sys::fuji_eval_execute(
            cache.c_code.as_ptr(), cache.c_code.len() as i32,
            cache.c_polys.as_ptr(), cache.c_polys.len() as i32,
            ctx.scalars.as_ptr() as *const fuji_sys::fuji_field,
            ctx.scalars.len() as i32,
            flat_scratch.as_mut_ptr() as *mut fuji_sys::fuji_field,
            scratch.len() as i32,
            ctx.chunk_start as i32, len as i32,
            &ctx.omega as *const FujiField as *const fuji_sys::fuji_field,
            CURVE as i32,
            out.as_mut_ptr() as *mut fuji_sys::fuji_field,
            std::ptr::null_mut(),
        );
    }
    for (i, row) in scratch.iter_mut().enumerate() {
        let base = i * len;
        row.copy_from_slice(&flat_scratch[base..base + len]);
    }
}

/// Serialize a compiled instruction sequence + context to .fuji bytecode format.
///
/// Returns the serialized bytes. Write to a file to get a portable .fuji.
pub fn save_bytecode(
    code: &[Instr],
    poly_table: &[PolyEntry],
    scalars: &[FujiField],
    omega: &FujiField,
    curve: FujiCurve,
    chunk_len: usize,
    max_depth: usize,
) -> Vec<u8> {
    let c_code: Vec<fuji_sys::fuji_eval_instr> = code.iter().map(|instr| {
        let (op, a0, a1) = match instr {
            Instr::LoadRotated { col, rotation } => (0, *col as i32, *rotation),
            Instr::Add => (1, 0, 0),
            Instr::Sub => (2, 0, 0),
            Instr::Mul => (3, 0, 0),
            Instr::Scale { scalar_idx } => (4, *scalar_idx as i32, 0),
            Instr::Fma { scalar_idx, acc_row } => (5, *scalar_idx as i32, *acc_row as i32),
            Instr::LinearTerm { scalar_idx } => (6, *scalar_idx as i32, 0),
            Instr::Constant { scalar_idx } => (7, *scalar_idx as i32, 0),
            Instr::Negate => (8, 0, 0),
            Instr::CopyToOut { src_row } => (9, *src_row as i32, 0),
            Instr::Dup => (10, 0, 0),
        };
        fuji_sys::fuji_eval_instr { opcode: op, arg0: a0, arg1: a1 }
    }).collect();

    let c_polys: Vec<fuji_sys::fuji_eval_poly> = poly_table.iter().map(|p| {
        fuji_sys::fuji_eval_poly {
            data: p.data.as_ptr() as *const fuji_sys::fuji_field,
            domain_size: p.domain_size as i32,
        }
    }).collect();

    let sz = unsafe {
        fuji_sys::fuji_bytecode_size(
            c_code.as_ptr(), c_code.len() as i32,
            c_polys.as_ptr(), c_polys.len() as i32,
            scalars.as_ptr() as *const fuji_sys::fuji_field, scalars.len() as i32,
            omega as *const FujiField as *const fuji_sys::fuji_field,
            curve as i32,
            chunk_len as i32, max_depth as i32,
        )
    };

    let mut buf = vec![0u8; sz];
    let rc = unsafe {
        fuji_sys::fuji_bytecode_write(
            c_code.as_ptr(), c_code.len() as i32,
            c_polys.as_ptr(), c_polys.len() as i32,
            scalars.as_ptr() as *const fuji_sys::fuji_field, scalars.len() as i32,
            omega as *const FujiField as *const fuji_sys::fuji_field,
            curve as i32,
            chunk_len as i32, max_depth as i32,
            buf.as_mut_ptr(), buf.len(),
        )
    };
    assert_eq!(rc, 0, "fuji_bytecode_write failed");
    buf
}

/// ── Bytecode wrapper — separates fixed structure from per-prove scalars ──
///
/// Loads a .fuji bytecode once (instructions + polynomials + metadata),
/// compiles the JIT once, then per prove just updates the challenge scalars
/// and dispatches. No recompilation, no re-encoding — ~450 ms per prove.

pub struct Bytecode {
    inner: *mut fuji_sys::fuji_bytecode,   // C-owned — instructions + poly + params
    fn_ptr: fuji_sys::jit_chunk_fn,          // compiled once
    scalars: Vec<FujiField>,                 // mutable — challenge slots patched each prove
    pub challenge_indices: Vec<usize>,       // which indices to update per prove
    n_polys: i32,
    omega: FujiField,
    curve: FujiCurve,
    pub chunk_len: i32,
    pub max_depth: i32,
    pub n_chunks: i32,
    pub n: i32,                              // domain size = n_chunks × chunk_len
    scs: Vec<Vec<fuji_sys::fuji_field>>,    // pre-allocated per-chunk scratch
    outs: Vec<Vec<fuji_sys::fuji_field>>,   // pre-allocated per-chunk output
    clones: Vec<fuji_sys::jit_chunk_fn>,    // per-thread MAP_JIT code copies
    clone_bufs: Vec<*mut libc::c_void>,     // raw mmap addresses (to munmap on free)
    clone_size: usize,                       // bytes per clone buffer
}

// The C bytecode handle is thread-safe (read-only after construction).
// The fn_ptr is a plain function pointer, always Send+Sync.
// The scalars/scs/outs are per-instance, not shared.
unsafe impl Send for Bytecode {}
unsafe impl Sync for Bytecode {}

impl Bytecode {
    /// Load a .fuji bytecode, compile JIT once, and prepare for per-prove execution.
    pub fn load(data: &[u8], challenge_indices: Vec<usize>) -> Self {
        let mut bc_ptr: *mut fuji_sys::fuji_bytecode = std::ptr::null_mut();
        unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
        let bc = unsafe { &*bc_ptr };

        let mut n_scalars: i32 = 0;
        let scalars_ptr = unsafe { fuji_sys::fuji_jit_bytecode_scalars(bc_ptr, &mut n_scalars) };
        let scalars = unsafe {
            std::slice::from_raw_parts(scalars_ptr as *const FujiField, n_scalars as usize)
                .to_vec()
        };
        let fn_ptr = unsafe { fuji_sys::fuji_jit_compile_from_bc(bc_ptr) };

        let cl = bc.chunk_len as usize;
        let md = bc.max_depth as usize;
        // Domain size from the first poly table entry (extended domain)
        let first_ds: usize = if bc.n_polys > 0 {
            unsafe { (*bc.poly_table).domain_size as usize }
        } else { 0 };
        let n = first_ds as i32;
        let num_chunks = (n as usize + cl - 1) / cl;
        let scratch_size = md * cl;

        // Pre-allocate per-chunk buffers once (no calloc/free per prove)
        let zero = fuji_sys::fuji_field { limbs: [0u8; 32] };
        let scs: Vec<Vec<fuji_sys::fuji_field>> = (0..num_chunks)
            .map(|_| vec![zero; scratch_size]).collect();
        let outs: Vec<Vec<fuji_sys::fuji_field>> = (0..num_chunks)
            .map(|_| vec![zero; cl]).collect();

        // Per-thread MAP_JIT clones. Execution of a shared MAP_JIT region is
        // serialized on M4 (per earlier investigation), so parallel dispatch
        // from a single code copy barely scales. Give each worker its own copy
        // of the code on its own physical pages via a single write-protect toggle.
        let n_threads = rayon::current_num_threads();
        let n_clones = std::cmp::min(n_threads, num_chunks).max(1);
        let code_size = unsafe { fuji_sys::fuji_jit_code_size() };
        let mut clones: Vec<fuji_sys::jit_chunk_fn> = Vec::with_capacity(n_clones);
        let mut clone_bufs: Vec<*mut libc::c_void> = Vec::with_capacity(n_clones);
        clones.push(fn_ptr);
        clone_bufs.push(std::ptr::null_mut());
        for _ in 1..n_clones {
            let buf = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    code_size,
                    libc::PROT_READ | libc::PROT_WRITE | libc::PROT_EXEC,
                    libc::MAP_ANONYMOUS | libc::MAP_PRIVATE | libc::MAP_JIT,
                    -1,
                    0,
                )
            };
            if buf == libc::MAP_FAILED {
                panic!("mmap MAP_JIT failed for JIT clone");
            }
            clone_bufs.push(buf);
            clones.push(unsafe { std::mem::transmute(buf) });
        }
        if n_clones > 1 {
            let rc = unsafe {
                fuji_sys::fuji_jit_batch_clone(
                    fn_ptr,
                    code_size,
                    clones[1..].as_mut_ptr(),
                    (n_clones - 1) as i32,
                )
            };
            if rc != 0 {
                panic!("fuji_jit_batch_clone failed with code {}", rc);
            }
        }

        Bytecode {
            inner: bc_ptr,
            fn_ptr,
            scalars,
            challenge_indices,
            n_polys: bc.n_polys,
            omega: unsafe { std::ptr::read(bc.omega as *const FujiField) },
            curve: match bc.curve { 0 => FujiCurve::Pallas, _ => FujiCurve::Vesta },
            chunk_len: bc.chunk_len,
            max_depth: bc.max_depth,
            n_chunks: num_chunks as i32,
            n,
            scs,
            outs,
            clones,
            clone_bufs,
            clone_size: code_size,
        }
    }

    /// Update the challenge scalar slots with fresh per-prove values.
    pub fn update_challenges(&mut self, fresh: &[FujiField]) {
        for (i, &idx) in self.challenge_indices.iter().enumerate() {
            if let Some(s) = fresh.get(i) {
                self.scalars[idx] = *s;
            }
        }
    }

    /// Execute all chunks using the pre-compiled JIT.
    ///
    /// `scalars` optionally overrides the embedded scalars from the `.fuji`
    /// bytecode. Pass `Some(&fresh_scalars)` when you compiled fresh challenge
    /// scalars per prove but want to reuse the cached instruction stream + JIT.
    /// Pass `None` to use the bytecode's embedded scalars.
    ///
    /// `polys` optionally overrides the embedded polynomial table. Polynomial
    /// data (permutation/lookup products) is recomputed per prove round, so
    /// callers must pass `Some(&fresh_polys)` when reusing a cached bytecode
    /// across rounds. Pass `None` to use the bytecode's embedded table.
    pub fn execute_all(&mut self, results: &mut [FujiField], n_threads: i32, scalars: Option<&[FujiField]>,
                       polys: Option<&[PolyEntry]>) {
        let _pd = std::env::var("PERF_DEBUG").is_ok();
        let _t0 = if _pd { Some(Instant::now()) } else { None };
        let num_chunks = self.n_chunks as usize;
        let chunk_len = self.chunk_len as usize;
        let zero = fuji_sys::fuji_field { limbs: [0u8; 32] };
        for sc in &mut self.scs { sc.fill(zero); }
        for out in &mut self.outs { out.fill(zero); }
        if let Some(ref _tn) = _t0 { eprintln!("[perf]   jit_zero_buf: {:.1}ms", _tn.elapsed().as_secs_f64() * 1000.0); }
        let _t1 = if _pd { Some(Instant::now()) } else { None };
        let bc = unsafe { &*self.inner };
        // Fresh C poly table (if overriding) — the owned Vec keeps the data
        // alive for the duration of this call.
        let fresh_c_polys: Vec<fuji_sys::fuji_eval_poly> = polys.map(|entries| {
            entries.iter().map(|p| fuji_sys::fuji_eval_poly {
                data: p.data.as_ptr() as *const fuji_sys::fuji_field,
                domain_size: p.domain_size as i32,
            }).collect()
        }).unwrap_or_default();
        let poly_table: *const fuji_sys::fuji_eval_poly = if polys.is_some() {
            fresh_c_polys.as_ptr()
        } else {
            bc.poly_table
        };
        let omega = &self.omega as *const FujiField as *const fuji_sys::fuji_field;
        let scalars_slice = scalars.unwrap_or(&self.scalars);
        let scalars = scalars_slice.as_ptr() as *const fuji_sys::fuji_field;
        let mod_ptr = unsafe { fuji_sys::fuji_field_modulus(self.curve as i32) };
        let p = mod_ptr as *const std::ffi::c_void;
        let pinv: u64 = match self.curve { FujiCurve::Pallas => 0x992d30ecffffffff, _ => 0x8c46eb20ffffffff };
        let mut sc_ptrs: Vec<*mut fuji_sys::fuji_field> =
            self.scs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        let mut out_ptrs: Vec<*mut fuji_sys::fuji_field> =
            self.outs.iter_mut().map(|v| v.as_mut_ptr()).collect();
        if let Some(ref _tn) = _t1 { eprintln!("[perf]   jit_setup: {:.1}ms", _tn.elapsed().as_secs_f64() * 1000.0); }
        let _t2 = if _pd { Some(Instant::now()) } else { None };
        if n_threads <= 0 {
            for ch in 0..num_chunks {
                unsafe {
                    (self.fn_ptr)(
                        (ch * chunk_len) as i32,
                        sc_ptrs[ch], out_ptrs[ch],
                        poly_table, scalars, omega, p, pinv,
                    );
                }
            }
        } else {
            // dispatch_apply_f — stable, ~748ms, no deadlocks.
            // fuji_jit_dispatch deadlocks when called from Rust on some M4 revisions.
            // Use the same extern "C" callback pattern that worked in testing.
            // Each chunk runs on a per-thread MAP_JIT clone so the M4 doesn't
            // serialize execution of a single shared code copy.
            let n_clones = self.clones.len();
            let clones = self.clones.as_ptr();
            extern "C" fn dispatch_chunk(ctx: *mut std::ffi::c_void, idx: usize) {
                let c = unsafe { &*(ctx as *const DispatchCtx) };
                let f = unsafe { *c.clones.add(idx % c.n_clones) };
                unsafe {
                    f(
                        (idx * c.chunk_len) as i32,
                        *c.scs.add(idx), *c.outs.add(idx),
                        c.poly_table, c.scalars, c.omega, c.p, c.pinv,
                    );
                }
            }
            let mut dctx = DispatchCtx {
                fn_ptr: self.fn_ptr,
                clones,
                n_clones,
                scs: sc_ptrs.as_mut_ptr(),
                outs: out_ptrs.as_mut_ptr(),
                poly_table, scalars, omega, p, pinv, chunk_len,
            };
            unsafe {
                dispatch_apply_f(
                    num_chunks,
                    dispatch_get_global_queue(0, 0),
                    &mut dctx as *mut _ as *mut std::ffi::c_void,
                    dispatch_chunk,
                );
            }
        }
        if let Some(ref _tn) = _t2 { eprintln!("[perf]   jit_dispatch: {:.1}ms", _tn.elapsed().as_secs_f64() * 1000.0); }
        let _t3 = if _pd { Some(Instant::now()) } else { None };
        for ch in 0..num_chunks {
            let base = ch * chunk_len;
            results[base..base + chunk_len]
                .copy_from_slice(unsafe {
                    std::mem::transmute::<&[fuji_sys::fuji_field], &[FujiField]>(&self.outs[ch])
                });
        }
        if let Some(ref _tn) = _t3 { eprintln!("[perf]   jit_copy_back: {:.1}ms", _tn.elapsed().as_secs_f64() * 1000.0); }
    }

    /// Free the bytecode and JIT buffer.
    pub fn free(self) {
        for &buf in &self.clone_bufs[1..] {
            if !buf.is_null() {
                unsafe { libc::munmap(buf, self.clone_size); }
            }
        }
        unsafe {
            fuji_sys::fuji_jit_free(self.fn_ptr);
            fuji_sys::fuji_bytecode_free(self.inner);
        }
    }
}

/// ── Thin eval binary IPC (shared file) ──────────────────
///
/// Writes scalars to a shared temp file in /tmp, spawns `fuji_eval`
/// to read the file (single big read) and write results to stdout.
/// The file is deleted after the child finishes.
/// This is faster than pipes because there's no context-switching
/// between parent and child during data transfer.

use std::io::Read;
use std::process::{Command, Stdio};
use std::io::Write;

pub struct EvalChild {
    eval_path: String,
    prove_path: String,
    n_scalars: usize,
    n_results: usize,
}

impl EvalChild {
    pub fn spawn(fuji_eval_path: &str, prove_path: &str, n_scalars: usize, n_results: usize) -> Self {
        EvalChild {
            eval_path: fuji_eval_path.to_owned(),
            prove_path: prove_path.to_owned(),
            n_scalars,
            n_results,
        }
    }

    /// Write scalars to a temp file, spawn child, read results.
    /// The child reads the file in one shot (fast, no pipe context-switching).
    pub fn eval(&self, scalars: &[u8], results: &mut [u8]) -> Result<(), String> {
        assert_eq!(scalars.len(), self.n_scalars * 32);
        assert_eq!(results.len(), self.n_results * 32);

        // Write scalars to a temp file
        let tmp_path = format!("/tmp/fuji_scalars.{}.bin", std::process::id());
        std::fs::write(&tmp_path, scalars)
            .map_err(|e| format!("write scalars: {}", e))?;

        let mut child = Command::new(&self.eval_path)
            .arg(&self.prove_path)
            .arg(&tmp_path)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("spawn: {}", e))?;

        let mut stdout = child.stdout.take()
            .ok_or_else(|| "no stdout".to_string())?;

        // Read results while the child is dispatching (overlaps with dispatch)
        stdout.read_exact(results)
            .map_err(|e| {
                let _ = std::fs::remove_file(&tmp_path);
                let mut stderr = String::new();
                if let Some(mut e) = child.stderr.take() {
                    let _ = e.read_to_string(&mut stderr);
                }
                format!("read stdout: {} (stderr: {})", e, stderr.trim())
            })?;

        let _ = std::fs::remove_file(&tmp_path);

        let status = child.wait()
            .map_err(|e| format!("wait: {}", e))?;
        if !status.success() {
            let mut stderr = String::new();
            if let Some(mut e) = child.stderr.take() {
                let _ = e.read_to_string(&mut stderr);
            }
            return Err(format!("exit={:?} stderr: {}", status.code(), stderr.trim()));
        }

        Ok(())
    }
}

// ── In-process JIT evaluator (no subprocess) ──────────────────
// Compile from a .fuji bytecode buffer once, then eval with new
// challenge scalars per prove. Uses dispatch_apply internally with
// USER_INTERACTIVE QoS for P-core preference on Apple Silicon.
// No fork, no pipes, no temp files — all in-process via FFI.

/// In-process JIT evaluator.
///
/// Compile once from a `.fuji` bytecode buffer, then call `eval`
/// with fresh challenge scalars per prove. The JIT is cached and
/// reused across calls — no recompilation, no subprocess overhead.
///
/// # Example
///
/// ```ignore
/// let bc_buf = std::fs::read("prove.fuji").unwrap();
/// let scalars = vec![FujiField::zero(); 5711];
/// let mut results = vec![FujiField::zero(); 16810];
///
/// let jit = JitEval::compile(&bc_buf, 5711, 16810).unwrap();
/// jit.eval(&scalars, &mut results).unwrap();
/// // jit dropped → cache freed
/// ```
pub struct JitEval {
    cache: *mut std::ffi::c_void,
    n_scalars: usize,
    n_results: usize,
}

unsafe impl Send for JitEval {}

impl JitEval {
    /// Compile a `.fuji` bytecode buffer and prepare for eval.
    ///
    /// `n_scalars` is the number of 32-byte challenge scalars each prove
    /// provides. `n_results` is the number of 32-byte result elements
    /// the evaluator produces (`num_chunks * chunk_len`).
    ///
    /// Returns `None` if the bytecode is invalid or JIT compilation fails.
    pub fn compile(bc_buf: &[u8], n_scalars: usize, n_results: usize) -> Option<Self> {
        let cache = unsafe { fuji_sys::fuji_jit_compile_from_buf(bc_buf.as_ptr(), bc_buf.len()) };
        if cache.is_null() {
            return None;
        }
        Some(JitEval { cache, n_scalars, n_results })
    }

    /// Evaluate all chunks with fresh challenge scalars.
    ///
    /// `scalars` must be exactly `n_scalars` field elements (each 32 bytes).
    /// `results` must be exactly `n_results` field elements.
    ///
    /// Returns an error string if the C eval failed (bad cache, internal error).
    pub fn eval(&self, scalars: &[FujiField], results: &mut [FujiField]) -> Result<(), String> {
        assert_eq!(scalars.len(), self.n_scalars, "scalars.len() != compile(n_scalars)");
        assert_eq!(results.len(), self.n_results, "results.len() != compile(n_results)");
        let rc = unsafe {
            fuji_sys::fuji_jit_eval_cached(
                self.cache,
                scalars.as_ptr() as *const fuji_sys::fuji_field,
                results.as_mut_ptr() as *mut fuji_sys::fuji_field,
            )
        };
        if rc != 0 {
            return Err(format!("fuji_jit_eval_cached returned {}", rc));
        }
        Ok(())
    }

    /// Evaluate all chunks with fresh challenge scalars and a fresh poly table.
    ///
    /// Permutation/lookup product polynomials are recomputed (and re-blinded)
    /// every prove round, so the bytecode's embedded poly table is stale for
    /// rounds 2+. Pass the current round's [`PolyEntry`] table here.
    pub fn eval_poly(&self, polys: &[PolyEntry], scalars: &[FujiField],
                     results: &mut [FujiField]) -> Result<(), String> {
        assert_eq!(scalars.len(), self.n_scalars, "scalars.len() != compile(n_scalars)");
        assert_eq!(results.len(), self.n_results, "results.len() != compile(n_results)");
        let c_polys: Vec<fuji_sys::fuji_eval_poly> = polys.iter().map(|p| {
            fuji_sys::fuji_eval_poly {
                data: p.data.as_ptr() as *const fuji_sys::fuji_field,
                domain_size: p.domain_size as i32,
            }
        }).collect();
        let rc = unsafe {
            fuji_sys::fuji_jit_eval_cached_poly(
                self.cache,
                c_polys.as_ptr(), c_polys.len() as i32,
                scalars.as_ptr() as *const fuji_sys::fuji_field,
                results.as_mut_ptr() as *mut fuji_sys::fuji_field,
            )
        };
        if rc != 0 {
            return Err(format!("fuji_jit_eval_cached_poly returned {}", rc));
        }
        Ok(())
    }
}

impl Drop for JitEval {
    fn drop(&mut self) {
        if !self.cache.is_null() {
            unsafe { fuji_sys::fuji_jit_cache_free(self.cache); }
        }
    }
}

/// Estimate the maximum scratch depth needed for a given instruction sequence.
/// Scans for the peak `sp` value assuming stack-like behavior.
pub fn estimate_depth(code: &[Instr]) -> usize {
    let mut sp = 0usize;
    let mut max_sp = 0usize;
    for instr in code {
        match instr {
            Instr::LoadRotated { .. }
            | Instr::LinearTerm { .. }
            | Instr::Constant { .. }
            | Instr::Dup => {
                sp += 1;
                max_sp = max_sp.max(sp);
            }
            Instr::Add | Instr::Sub | Instr::Mul => {
                sp = sp.saturating_sub(1);
            }
            Instr::Fma { acc_row, .. } => {
                if *acc_row != 0xFFFFFFFF {
                    sp = *acc_row as usize + 1;  // acc_row mode: reset sp above accumulator
                } else {
                    sp = sp.saturating_sub(1);    // legacy: sp-2 accumulator
                }
            }
            Instr::Scale { .. } | Instr::Negate => { /* sp unchanged */ }
            Instr::CopyToOut { .. } => { /* no change */ }
        }
    }
    // Must accommodate chain dispatch: sp + n_terms + 1 temp row.
    // n_terms can be up to MAX_TERMS = 16 in the C pre-scanner.
    max_sp + 18
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_exec(code: &[Instr], poly_table: &[PolyEntry], scalars: &[FujiField],
                omega: &FujiField, chunk_len: usize, max_depth: usize, n: usize) -> Vec<FujiField>
    {
        let n_chunks = (n + chunk_len - 1) / chunk_len;
        let mut results = vec![FujiField::zero(); n_chunks * chunk_len];
        let zero = FujiField::zero();
        for ch in 0..n_chunks {
            let start = ch * chunk_len;
            let mut scratch = vec![vec![zero; chunk_len]; max_depth.max(3)];
            let mut sp = 0usize;
            for instr in code {
                match instr {
                    Instr::LoadRotated { col, rotation } => {
                        let p = &poly_table[*col as usize];
                        for i in 0..chunk_len {
                            let idx = if *rotation >= 0 {
                                (start + i + *rotation as usize) % p.domain_size
                            } else {
                                let a = (-rotation) as usize;
                                if start + i >= a { start + i - a }
                                else { p.domain_size - (a - (start + i)) % p.domain_size }
                            };
                            scratch[sp][i] = p.data[idx % p.domain_size];
                        }
                        sp += 1;
                    }
                    Instr::Add => { let (a,b) = (sp-2,sp-1); let ta = scratch[a].clone(); let tb = scratch[b].clone();
                        crate::batch_field::add(&ta, &tb, &mut scratch[a], CURVE); sp-=1; }
                    Instr::Sub => { let (a,b) = (sp-2,sp-1); let ta = scratch[a].clone(); let tb = scratch[b].clone();
                        crate::batch_field::sub(&ta, &tb, &mut scratch[a], CURVE); sp-=1; }
                    Instr::Mul => { let (a,b) = (sp-2,sp-1); let ta = scratch[a].clone(); let tb = scratch[b].clone();
                        crate::batch_field::mul(&ta, &tb, &mut scratch[a], CURVE); sp-=1; }
                    Instr::Scale { scalar_idx } => { let s = &scalars[*scalar_idx as usize];
                        let t = scratch[sp-1].clone();
                        crate::batch_field::scale(&t, s, &mut scratch[sp-1], CURVE); }
                    Instr::Fma { scalar_idx, acc_row } => { let s = &scalars[*scalar_idx as usize];
                        let a = if *acc_row != 0xFFFFFFFF { *acc_row as usize } else { sp - 2 };
                        let term = sp - 1;
                        let ta = scratch[a].clone(); let tterm = scratch[term].clone();
                        crate::batch_field::scale(&ta, s, &mut scratch[a], CURVE);
                        let ta2 = scratch[a].clone();
                        crate::batch_field::add(&ta2, &tterm, &mut scratch[a], CURVE);
                        sp = if *acc_row != 0xFFFFFFFF { *acc_row as usize + 1 } else { sp - 1 }; }
                    Instr::LinearTerm { scalar_idx } => {
                        let s = &scalars[*scalar_idx as usize];
                        // pow = ω^start, starting from Mont(1)
                        let mut one = [0u8; 32]; one[0] = 1;
                        let mut pow = FujiField::from_bytes(&one).to_mont(CURVE);
                        for _ in 0..start {
                            let src = pow;
                            let sc = pow;
                            crate::batch_field::scale(&[src], &sc, std::slice::from_mut(&mut pow), CURVE);
                        }
                        for i in 0..chunk_len {
                            if i == 0 {
                                crate::batch_field::scale(&[*s], &pow, &mut scratch[sp][0..1], CURVE);
                            } else {
                                let prev = scratch[sp][i - 1];
                                crate::batch_field::mul(&[prev], &[*omega], &mut scratch[sp][i..i + 1], CURVE);
                            }
                        }
                        sp += 1;
                    }
                    Instr::Constant { scalar_idx } => { scratch[sp].fill(scalars[*scalar_idx as usize]); sp+=1; }
                    Instr::Negate => { let r = sp-1; let z = vec![zero; chunk_len]; let t = scratch[r].clone();
                        crate::batch_field::sub(&z, &t, &mut scratch[r], CURVE); }
                    Instr::CopyToOut { src_row } => { let src = if *src_row != 0xFFFFFFFF { *src_row as usize } else { 0 };
                        results[ch*chunk_len..(ch+1)*chunk_len].copy_from_slice(&scratch[src][..chunk_len]); }
                    Instr::Dup => { let s = scratch[sp-1].clone(); scratch[sp][..chunk_len].copy_from_slice(&s); sp+=1; }
                }
            }
        }
        results
    }

    fn make_poly(data: &[u64]) -> Vec<FujiField> {
        data.iter().map(|&v| { let mut b = [0u8; 32]; b[..8].copy_from_slice(&v.to_le_bytes()); FujiField::from_bytes(&b) }).collect()
    }

    fn run_opcode(code: Vec<Instr>, mut poly_table: Vec<PolyEntry>, scalars: Vec<FujiField>,
                  n: usize, chunk_len: usize, desc: &str) {
        let omega = FujiField::from_bytes(&{ let mut b = [0u8; 32]; b[0] = 5; b });
        let md = estimate_depth(&code);
        // Ensure at least one poly entry so Bytecode::load computes n correctly
        if poly_table.is_empty() {
            poly_table.push(PolyEntry{ data: make_poly(&[0u64; 10]), domain_size: n });
        }
        // Align bc_bytes to 16 bytes (misaligned poly_table pointer crashes)
        let raw = save_bytecode(&code, &poly_table, &scalars, &omega, CURVE, chunk_len, md);
        let aligned_len = ((raw.len() + 15) / 16) * 16;
        let mut aligned = vec![0u8; aligned_len];
        aligned[..raw.len()].copy_from_slice(&raw);
        let mut bc = Bytecode::load(&aligned, vec![]);
        let bc_n = bc.n as usize;
        let mut jit = vec![FujiField::zero(); bc_n];
        bc.execute_all(&mut jit, 0, Some(&scalars), None);
        drop(bc);
        let ref_ = ref_exec(&code, &poly_table, &scalars, &omega, chunk_len, md, bc_n);
        let mut diffs = 0;
        for (i,(j,r)) in jit.iter().zip(ref_.iter()).enumerate() {
            if j != r && diffs < 5 { eprintln!("  [{desc}] mismatch [{i}]: jit=0x{:02x?} ref=0x{:02x?}", j.to_bytes(), r.to_bytes()); diffs+=1; }
        }
        assert_eq!(diffs, 0, "{desc}: {diffs} mismatches");
    }

    #[test] fn op_add() {
        let p = vec![PolyEntry{data:make_poly(&[1,2,3,4,5,6,7,8,9,10]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::LoadRotated{col:0,rotation:0},Instr::Add,Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "ADD");
    }
    #[test] fn op_mul() {
        let p = vec![PolyEntry{data:make_poly(&[3,5,7,9,11,13,15,17,19,21]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::LoadRotated{col:0,rotation:0},Instr::Mul,Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "MUL");
    }
    #[test] fn op_scale() {
        let p = vec![PolyEntry{data:make_poly(&[2,4,6,8,10,12,14,16,18,20]),domain_size:10}];
        let s = vec![FujiField::from_bytes(&{let mut b=[0u8;32];b[0]=3;b})];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::Scale{scalar_idx:0},Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, s, 10, 10, "SCALE");
    }
    #[test] fn op_fma() {
        let p = vec![PolyEntry{data:make_poly(&[1,2,3,4,5,6,7,8,9,10]),domain_size:10}];
        let s = vec![FujiField::from_bytes(&{let mut b=[0u8;32];b[0]=2;b})];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::LoadRotated{col:0,rotation:0},Instr::Fma{scalar_idx:0, acc_row:0xFFFFFFFF},Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, s, 10, 10, "FMA");
    }
    #[test] fn op_negate() {
        let p = vec![PolyEntry{data:make_poly(&[7,8,9,10,11,12,13,14,15,16]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::Negate,Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "NEGATE");
    }
    #[test] fn op_constant() {
        let s = vec![FujiField::from_bytes(&{let mut b=[0u8;32];b[0]=42;b})];
        run_opcode(vec![Instr::Constant{scalar_idx:0},Instr::CopyToOut { src_row: 0xFFFFFFFF }], vec![], s, 10, 10, "CONSTANT");
    }
    #[test] fn op_linear_term() {
        let s = vec![FujiField::from_bytes(&{let mut b=[0u8;32];b[0]=1;b})];
        run_opcode(vec![Instr::LinearTerm{scalar_idx:0},Instr::CopyToOut { src_row: 0xFFFFFFFF }], vec![], s, 10, 10, "LINEAR_TERM");
    }
    #[test] fn op_load_rotated() {
        let p = vec![PolyEntry{data:make_poly(&[100,101,102,103,104,105,106,107,108,109]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:2},Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "LOAD_ROTATED");
    }
    #[test] fn op_load_rotated_neg() {
        let p = vec![PolyEntry{data:make_poly(&[100,101,102,103,104,105,106,107,108,109]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:-2},Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "LOAD_ROTATED_NEG");
    }
    #[test] fn op_sub() {
        let p = vec![PolyEntry{data:make_poly(&[10,9,8,7,6,5,4,3,2,1]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::LoadRotated{col:0,rotation:0},Instr::Sub,Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "SUB");
    }
    #[test] fn op_dup() {
        let p = vec![PolyEntry{data:make_poly(&[5,6,7,8,9,10,11,12,13,14]),domain_size:10}];
        run_opcode(vec![Instr::LoadRotated{col:0,rotation:0},Instr::Dup,Instr::Add,Instr::CopyToOut { src_row: 0xFFFFFFFF }], p, vec![], 10, 10, "DUP");
    }
    #[test] fn op_all_ops_stacked() {
        let p = vec![PolyEntry{data:make_poly(&[1,2,3,4,5,6,7,8,9,10]),domain_size:10}];
        let s = vec![FujiField::from_bytes(&{let mut b=[0u8;32];b[0]=3;b})];
        // (poly + poly) * poly / ... just a mul after add
        run_opcode(vec![
            Instr::LoadRotated{col:0,rotation:0},
            Instr::LoadRotated{col:0,rotation:0},
            Instr::Mul,
            Instr::Constant{scalar_idx:0},
            Instr::Sub,
            Instr::CopyToOut { src_row: 0xFFFFFFFF },
        ], p, s, 10, 10, "ALL_OPS");
    }
}
