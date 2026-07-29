use std::{
    any::TypeId,
    cell::RefCell,
    cmp, fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, Mul, MulAssign, Neg, Sub},
    sync::Arc,
};
use std::sync::atomic::{AtomicU64, Ordering};

use ff::WithSmallOrderMulGroup;
use group::ff::Field;

use super::{
    Basis, Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, Rotation,
};
use crate::multicore;

#[cfg(feature = "fuji")]
use fuji as fuji_crate;
#[cfg(feature = "fuji")]
use crate::arithmetic::fuji as fuji_helpers;

/// Returns `(chunk_size, num_chunks)` suitable for processing the given polynomial length
/// in the current parallelization environment.
fn get_chunk_params(poly_len: usize) -> (usize, usize) {
    // Check the level of parallelization we have available.
    let num_threads = multicore::current_num_threads();
    // We scale the number of chunks by a constant factor, to ensure that if not all
    // threads are available, we can achieve more uniform throughput and don't end up
    // waiting on a couple of threads to process the last chunks.
    let num_chunks = num_threads * 4;
    // Calculate the ideal chunk size for the desired throughput. We use ceiling
    // division to ensure the minimum chunk size is 1.
    //     chunk_size = ceil(poly_len / num_chunks)
    let chunk_size = (poly_len + num_chunks - 1) / num_chunks;
    // Now re-calculate num_chunks from the actual chunk size.
    //     num_chunks = ceil(poly_len / chunk_size)
    let num_chunks = (poly_len + chunk_size - 1) / chunk_size;

    (chunk_size, num_chunks)
}

/// A reference to a polynomial registered with an [`Evaluator`].
#[derive(Clone, Copy)]
pub(crate) struct AstLeaf<E, B: Basis> {
    index: usize,
    rotation: Rotation,
    _evaluator: PhantomData<(E, B)>,
}

impl<E, B: Basis> fmt::Debug for AstLeaf<E, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AstLeaf")
            .field("index", &self.index)
            .field("rotation", &self.rotation)
            .finish()
    }
}

impl<E, B: Basis> PartialEq for AstLeaf<E, B> {
    fn eq(&self, rhs: &Self) -> bool {
        // We compare rotations by offset, which doesn't account for equivalent rotations.
        self.index.eq(&rhs.index) && self.rotation.0.eq(&rhs.rotation.0)
    }
}

impl<E, B: Basis> Eq for AstLeaf<E, B> {}

impl<E, B: Basis> Hash for AstLeaf<E, B> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.rotation.0.hash(state);
    }
}

impl<E, B: Basis> AstLeaf<E, B> {
    /// Produces a new `AstLeaf` node corresponding to the underlying polynomial at a
    /// _new_ rotation. Existing rotations applied to this leaf node are ignored and the
    /// returned polynomial is not rotated _relative_ to the previous structure.
    pub(crate) fn with_rotation(&self, rotation: Rotation) -> Self {
        AstLeaf {
            index: self.index,
            rotation,
            _evaluator: PhantomData,
        }
    }
}

/// An evaluation context for polynomial operations.
///
/// This context enables us to de-duplicate queries of circuit columns (and the rotations
/// they might require), by storing a list of all the underlying polynomials involved in
/// any query (which are almost certainly column polynomials). We use the context like so:
///
/// - We register each underlying polynomial with the evaluator, which returns a reference
///   to it as a [`AstLeaf`].
/// - The references are then used to build up a [`Ast`] that represents the overall
///   operations to be applied to the polynomials.
/// - Finally, we call [`Evaluator::evaluate`] passing in the [`Ast`].
pub(crate) struct Evaluator<E, F: Field, B: Basis> {
    polys: Vec<Polynomial<F, B>>,
    _context: E,
    #[cfg(feature = "fuji")]
    fuji_curve: Option<fuji_crate::FujiCurve>,
}

/// Constructs a new `Evaluator`.
///
/// The `context` parameter is used to provide type safety for evaluators. It ensures that
/// an evaluator will only be used to evaluate [`Ast`]s containing [`AstLeaf`]s obtained
/// from itself. It should be set to the empty closure `|| {}`, because anonymous closures
/// all have unique types.
pub(crate) fn new_evaluator<E: Fn() + Clone, F: Field, B: Basis>(context: E) -> Evaluator<E, F, B> {
    Evaluator {
        polys: vec![],
        _context: context,
        #[cfg(feature = "fuji")]
        fuji_curve: None,
    }
}

impl<E, F: Field, B: Basis + 'static> Evaluator<E, F, B> {
    /// Registers the given polynomial for use in this evaluation context.
    ///
    /// This API treats each registered polynomial as unique, even if the same polynomial
    /// is added multiple times.
    pub(crate) fn register_poly(&mut self, poly: Polynomial<F, B>) -> AstLeaf<E, B> {
        let index = self.polys.len();
        self.polys.push(poly);

        AstLeaf {
            index,
            rotation: Rotation::cur(),
            _evaluator: PhantomData,
        }
    }

    /// Enable the Fuji SME-accelerated flat evaluator path.
    #[cfg(feature = "fuji")]
    pub(crate) fn enable_fuji(&mut self, curve: fuji_crate::FujiCurve) {
        self.fuji_curve = Some(curve);
    }

        // Evaluates the given polynomial operation against this context.
    pub(crate) fn evaluate(
        &self,
        ast: &Ast<E, F, B>,
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, B>
    where
        E: Copy + Send + Sync,
        F: WithSmallOrderMulGroup<3> + ff::PrimeField,
        B: BasisOps,
    {
        #[cfg(feature = "fuji")]
        if let Some(ref curve) = self.fuji_curve {
            if fuji_helpers::fuji_available() && TypeId::of::<B>() == TypeId::of::<ExtendedLagrangeCoeff>() {
                return self.evaluate_fuji(ast, domain, *curve);
            }
        }

        // We're working in a single basis, so all polynomials are the same length.
        let poly_len = self.polys.first().unwrap().len();
        let (chunk_size, _num_chunks) = get_chunk_params(poly_len);

        struct AstContext<'a, F: Field, B: Basis> {
            domain: &'a EvaluationDomain<F>,
            chunk_size: usize,
            chunk_index: usize,
            polys: &'a [Polynomial<F, B>],
        }

        #[inline(always)]
        fn count(counters: &[AtomicU64; 7], idx: usize) {
            if std::env::var("PERF_DEBUG").is_ok() {
                counters[idx].fetch_add(1, Ordering::Relaxed);
            }
        }

        fn recurse_into<E, F: WithSmallOrderMulGroup<3>, B: BasisOps>(
            out: &mut [F],
            ast: &Ast<E, F, B>,
            ctx: &AstContext<'_, F, B>,
            stack: &mut [Vec<F>],
            counters: &[AtomicU64; 7],
            timers: &[AtomicU64; 7],
        ) {
            match ast {
                Ast::Poly(leaf) => {
                    count(counters, 0);
                    let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                    B::get_chunk_of_rotated_into(
                        out,
                        ctx.domain,
                        ctx.chunk_size,
                        ctx.chunk_index,
                        &ctx.polys[leaf.index],
                        leaf.rotation,
                    );
                    if let Some(ref t) = _t { timers[0].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                }
                Ast::Add(a, b) => {
                    count(counters, 1);
                    recurse_into(out, b, ctx, stack, counters, timers);
                    let (first, rest) = stack.split_at_mut(1);
                    recurse_into(&mut first[0], a, ctx, rest, counters, timers);
                    let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                    for i in 0..out.len() {
                        out[i] += first[0][i];
                    }
                    if let Some(ref t) = _t { timers[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                }
                Ast::Mul(AstMul(a, b)) => {
                    count(counters, 2);
                    recurse_into(out, b, ctx, stack, counters, timers);
                    let (first, rest) = stack.split_at_mut(1);
                    recurse_into(&mut first[0], a, ctx, rest, counters, timers);
                    let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                    for i in 0..out.len() {
                        out[i] *= first[0][i];
                    }
                    if let Some(ref t) = _t { timers[2].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                }
                Ast::Scale(a, scalar) => {
                    count(counters, 3);
                    recurse_into(out, a, ctx, stack, counters, timers);
                    let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                    for lhs in out.iter_mut() {
                        *lhs *= scalar;
                    }
                    if let Some(ref t) = _t { timers[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                }
                Ast::DistributePowers(terms, base) => {
                    count(counters, 4);
                    let mut terms = terms.iter();
                    if let Some(first_term) = terms.next() {
                        recurse_into(out, first_term, ctx, stack, counters, timers);
                        for term in terms {
                            let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                            for elem in out.iter_mut() {
                                *elem *= base;
                            }
                            if let Some(ref t) = _t { timers[3].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); } // Scale portion
                            let (first, rest) = stack.split_at_mut(1);
                            recurse_into(&mut first[0], term, ctx, rest, counters, timers);
                            let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                            for i in 0..out.len() {
                                out[i] += first[0][i];
                            }
                            if let Some(ref t) = _t { timers[1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); } // Add portion
                        }
                    } else {
                        out.fill(F::ZERO);
                    }
                }
                Ast::LinearTerm(scalar) => {
                    count(counters, 5);
                    let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                    B::linear_term_into(out, ctx.domain, ctx.chunk_size, ctx.chunk_index, *scalar);
                    if let Some(ref t) = _t { timers[5].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                }
                Ast::ConstantTerm(scalar) => {
                    count(counters, 6);
                    let _t = if std::env::var("PERF_DEBUG").is_ok() { let n = std::time::Instant::now(); Some(n) } else { None };
                    B::constant_term_into(out, *scalar);
                    if let Some(ref t) = _t { timers[6].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed); }
                }
            }
        }

        // Apply `ast` to each chunk in parallel, writing the result into an output
        // polynomial.
        let _t_compile = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let _depth = ast.depth();
        if let Some(ref tc) = _t_compile { eprintln!("[perf]   ast_compile_depth: {:.1}ms depth={}", tc.elapsed().as_secs_f64() * 1000.0, _depth); }

        let _t_eval = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let counters: [AtomicU64; 7] = [
            AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            AtomicU64::new(0),
        ];
        let timers: [AtomicU64; 7] = [
            AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            AtomicU64::new(0),
        ];
        let counters_ref = &counters;
        let timers_ref = &timers;
        let depth = _depth;
        let mut result = B::empty_poly(domain);
        multicore::scope(|scope| {
            for (chunk_index, out) in result.chunks_mut(chunk_size).enumerate() {
                scope.spawn(move |_| {
                    let ctx = AstContext {
                        domain,
                        chunk_size,
                        chunk_index,
                        polys: &self.polys,
                    };
                    let mut scratch: Vec<Vec<F>> = (0..depth + 1)
                        .map(|_| vec![F::ZERO; out.len()])
                        .collect();
                    recurse_into(out, ast, &ctx, &mut scratch[..], counters_ref, timers_ref);
                });
            }
        });
        if std::env::var("PERF_DEBUG").is_ok() {
            let c = |i: usize| counters[i].load(Ordering::Relaxed);
            let t = |i: usize| timers[i].load(Ordering::Relaxed) as f64 / 1_000_000.0;
            let total_ns = timers.iter().map(|t| t.load(Ordering::Relaxed)).sum::<u64>() as f64;
            eprintln!(
                "[perf] AST: Poly={} Add={} Mul={} Scale={} DP={} Lin={} Const={}",
                c(0), c(1), c(2), c(3), c(4), c(5), c(6)
            );
            eprintln!(
                "[perf] AST_NS: Poly={:.1} Add={:.1} Mul={:.1} Scale={:.1} DP={:.1} Lin={:.1} Const={:.1}  (total self={:.1}ms wall={:.1}ms)",
                t(0), t(1), t(2), t(3), t(4), t(5), t(6),
                total_ns / 1_000_000.0,
                _t_eval.as_ref().unwrap().elapsed().as_secs_f64() * 1000.0,
            );
        }
        result
    }

    /// Evaluate using Fuji's flat-instruction SME evaluator.
    #[cfg(feature = "fuji")]
    fn evaluate_fuji(
        &self,
        ast: &Ast<E, F, B>,
        domain: &EvaluationDomain<F>,
        curve: fuji_crate::FujiCurve,
    ) -> Polynomial<F, B>
    where
        F: WithSmallOrderMulGroup<3> + ff::PrimeField,
        B: BasisOps,
    {

        let poly_len = self.polys.first().unwrap().len();
        let domain_n = 1 << domain.k();
        let stride = poly_len / domain_n;
        let chunk_len = 410;

        // Thread-local Bytecode cache — compiled once per thread. dispatch_apply_f inside.
        // JitEval is slower on current library (938ms vs 765ms Bytecode).
        #[cfg(feature = "fuji")]
        thread_local! {
            static BC_CACHE: RefCell<Option<fuji_crate::eval::Bytecode>> = const { RefCell::new(None) };
        }

        let omega_native = domain.get_extended_omega();
        let omega_mont = {
            let mut buf = [0u8; 32];
            let repr = omega_native.to_repr();
            let bytes: &[u8] = repr.as_ref();
            buf[..bytes.len()].copy_from_slice(bytes);
            fuji_crate::FujiField::from_bytes(&buf).to_mont(curve)
        };

        // 2. Build or load cached Bytecode
        let n_padded = ((poly_len + chunk_len - 1) / chunk_len) * chunk_len;
        BC_CACHE.with(|cache| {
            let mut guard = cache.borrow_mut();
            if guard.is_none() {
                let (code, scalars, ci) =
                    compile_ast(ast, stride as u32, omega_native, curve);
                let max_depth = fuji_crate::eval::estimate_depth(&code);
                let poly_table: Vec<fuji_crate::eval::PolyEntry> = self.polys.iter().map(|poly| {
                    let data = poly.values.iter().map(|f| {
                        let mut buf = [0u8; 32];
                        let repr = f.to_repr();
                        let bytes: &[u8] = repr.as_ref();
                        buf[..bytes.len()].copy_from_slice(bytes);
                        fuji_crate::FujiField::from_bytes(&buf).to_mont(curve)
                    }).collect();
                    fuji_crate::eval::PolyEntry { data, domain_size: poly.values.len() }
                }).collect();

                let bc_bytes = fuji_crate::eval::save_bytecode(
                    &code, &poly_table, &scalars, &omega_mont, curve, chunk_len, max_depth,
                );
                if std::env::var("PERF_DEBUG").is_ok() || std::env::var("FUJI_DUMP_BC").is_ok() {
                    use std::io::Write;
                    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
                    if std::env::var("FUJI_DUMP_BC").is_ok() {
                        let fname = format!("prove_{}.fuji", ts);
                        let _ = (|| -> std::io::Result<()> { let mut f = std::fs::File::create(&fname)?; f.write_all(&bc_bytes) })();
                    }
                    if std::env::var("PERF_DEBUG").is_ok() {
                        let fname = format!("ast_dump_{}.txt", ts);
                        if let Ok(mut f) = std::fs::File::create(&fname) {
                            let _ = writeln!(f, ";; ast_dump  stride={}  chunk_len={}  poly_count={}  scalar_count={}",
                                stride, chunk_len, poly_table.len(), scalars.len());
                            for instr in &code {
                                let _ = writeln!(f, "{}", fmt_instr(instr, &scalars));
                            }
                        }
                    }
                }
                if std::env::var("FUJI_TRACE_AST").is_ok() {
                    let trace_ctx = fuji_crate::eval::EvalContext {
                        chunk_len, chunk_start: 0, max_depth,
                        poly_table: poly_table.clone(), omega: omega_mont, scalars: scalars.clone(),
                    };
                    eprintln!(";; trace_execute  stride={}  chunk_len={}  poly_count={}  scalar_count={}  max_depth={}",
                        stride, chunk_len, trace_ctx.poly_table.len(), trace_ctx.scalars.len(), trace_ctx.max_depth);
                    trace_execute(&code, &trace_ctx, curve);
                }
                *guard = Some(fuji_crate::eval::Bytecode::load(&bc_bytes, ci));
            }
        });

        // 3. Execute via cached Bytecode
        let _jit_timer = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let mut results_mont = vec![fuji_crate::FujiField::zero(); n_padded];
        BC_CACHE.with(|cache| {
            cache.borrow_mut().as_mut().unwrap().execute_all(&mut results_mont, 3);
        });
        if let Some(ref t) = _jit_timer { eprintln!("[perf]   jit_eval: {:.1}ms", t.elapsed().as_secs_f64() * 1000.0); }

        // 4. Convert from Montgomery to native field
        let mut result_vals = Vec::with_capacity(poly_len);
        for val in &results_mont[..poly_len] {
            let nv = val.from_mont(curve);
            let bytes = nv.to_bytes();
            let mut repr = F::Repr::default();
            let dst: &mut [u8] = repr.as_mut();
            dst.copy_from_slice(&bytes);
            result_vals.push(F::from_repr(repr).unwrap());
        }

        Polynomial {
            values: result_vals,
            _marker: PhantomData,
        }
    }
}

/// Compile a recursive AST (ExtendedLagrangeCoeff basis) into a flat
/// `Vec<fuji_crate::eval::Instr>` plus a scalar table. All scalars are
/// emitted in Montgomery form.
#[cfg(feature = "fuji")]
fn compile_ast<Ev, F: WithSmallOrderMulGroup<3>, B: Basis>(
    ast: &Ast<Ev, F, B>,
    stride: u32,
    omega_native: F,
    curve: fuji_crate::FujiCurve,
) -> (Vec<fuji_crate::eval::Instr>, Vec<fuji_crate::FujiField>, Vec<usize>) {
    let mut code = Vec::new();
    let mut scalars = Vec::new();
    let mut challenge_indices = Vec::new();
    compile_node(ast, &mut code, &mut scalars, &mut challenge_indices, stride, omega_native, curve);
    code.push(fuji_crate::eval::Instr::CopyToOut);
    (code, scalars, challenge_indices)
}

#[cfg(feature = "fuji")]
fn push_scalar<F: ff::PrimeField + Field>(
    scalars: &mut Vec<fuji_crate::FujiField>,
    f: F,
    curve: fuji_crate::FujiCurve,
) -> u32 {
    let mut buf = [0u8; 32];
    let repr = f.to_repr();
    let bytes: &[u8] = repr.as_ref();
    buf[..bytes.len()].copy_from_slice(bytes);
    let normal = fuji_crate::FujiField::from_bytes(&buf);
    let mont = normal.to_mont(curve);
    let idx = scalars.len() as u32;
    scalars.push(mont);
    idx
}

#[cfg(feature = "fuji")]
fn compile_node<E, F: WithSmallOrderMulGroup<3>, B: Basis>(
    ast: &Ast<E, F, B>,
    code: &mut Vec<fuji_crate::eval::Instr>,
    scalars: &mut Vec<fuji_crate::FujiField>,
    challenge_indices: &mut Vec<usize>,
    stride: u32,
    omega_native: F,
    curve: fuji_crate::FujiCurve,
) {
    match ast {
        Ast::Poly(leaf) => {
            let elt_rotation = leaf.rotation.0 * stride as i32;
            code.push(fuji_crate::eval::Instr::LoadRotated {
                col: leaf.index as u32,
                rotation: elt_rotation,
            });
        }
        Ast::Add(a, b) => {
            compile_node(b, code, scalars, challenge_indices, stride, omega_native, curve);
            compile_node(a, code, scalars, challenge_indices, stride, omega_native, curve);
            code.push(fuji_crate::eval::Instr::Add);
        }
        Ast::Mul(AstMul(a, b)) => {
            compile_node(b, code, scalars, challenge_indices, stride, omega_native, curve);
            compile_node(a, code, scalars, challenge_indices, stride, omega_native, curve);
            code.push(fuji_crate::eval::Instr::Mul);
        }
        Ast::Scale(a, scalar) => {
            compile_node(a, code, scalars, challenge_indices, stride, omega_native, curve);
            let idx = push_scalar(scalars, *scalar, curve);
            code.push(fuji_crate::eval::Instr::Scale { scalar_idx: idx });
        }
        Ast::DistributePowers(terms, base) => {
            let mut terms = terms.iter();
            if let Some(first) = terms.next() {
                compile_node(first, code, scalars, challenge_indices, stride, omega_native, curve);
                for term in terms {
                    compile_node(term, code, scalars, challenge_indices, stride, omega_native, curve);
                    let idx = push_scalar(scalars, *base, curve);
                    challenge_indices.push(idx as usize);
                    code.push(fuji_crate::eval::Instr::Fma { scalar_idx: idx });
                }
            } else {
                let idx = push_scalar(scalars, F::ZERO, curve);
                code.push(fuji_crate::eval::Instr::Constant { scalar_idx: idx });
            }
        }
        Ast::LinearTerm(scalar) => {
            let zeta_scaled = *scalar * F::ZETA;
            let idx = push_scalar(scalars, zeta_scaled, curve);
            challenge_indices.push(idx as usize);
            code.push(fuji_crate::eval::Instr::LinearTerm { scalar_idx: idx });
        }
        Ast::ConstantTerm(scalar) => {
            let idx = push_scalar(scalars, *scalar, curve);
            code.push(fuji_crate::eval::Instr::Constant { scalar_idx: idx });
        }
    }
}

#[cfg(feature = "fuji")]
fn fmt_instr(instr: &fuji_crate::eval::Instr, scalars: &[fuji_crate::FujiField]) -> String {
    match instr {
        fuji_crate::eval::Instr::LoadRotated { col, rotation } => {
            format!("LOAD_ROTATED col={} rot={}", col, rotation)
        }
        fuji_crate::eval::Instr::Add => "ADD".into(),
        fuji_crate::eval::Instr::Sub => "SUB".into(),
        fuji_crate::eval::Instr::Mul => "MUL".into(),
        fuji_crate::eval::Instr::Scale { scalar_idx } => {
            let s = &scalars[*scalar_idx as usize];
            format!("SCALE scalar_idx={} val=0x{:02x?}", scalar_idx, s.0)
        }
        fuji_crate::eval::Instr::Fma { scalar_idx } => {
            let s = &scalars[*scalar_idx as usize];
            format!("FMA scalar_idx={} val=0x{:02x?}", scalar_idx, s.0)
        }
        fuji_crate::eval::Instr::LinearTerm { scalar_idx } => {
            let s = &scalars[*scalar_idx as usize];
            format!("LINEAR_TERM scalar_idx={} val=0x{:02x?}", scalar_idx, s.0)
        }
        fuji_crate::eval::Instr::Constant { scalar_idx } => {
            let s = &scalars[*scalar_idx as usize];
            format!("CONSTANT scalar_idx={} val=0x{:02x?}", scalar_idx, s.0)
        }
        fuji_crate::eval::Instr::Negate => "NEGATE".into(),
        fuji_crate::eval::Instr::CopyToOut => "COPY_TO_OUT".into(),
        fuji_crate::eval::Instr::Dup => "DUP".into(),
    }
}

/// Trace-execute the compiled instruction sequence on chunk 0, printing every
/// operation together with the first few field element values it produces.
/// Triggered by `FUJI_TRACE_AST=1`.
#[cfg(feature = "fuji")]
fn trace_execute(code: &[fuji_crate::eval::Instr], ctx: &fuji_crate::eval::EvalContext, curve: fuji_crate::FujiCurve) {
    use fuji_crate::batch_field;
    let n = ctx.chunk_len;
    let n_rows = ctx.max_depth.max(3); // at least 3 for temp row
    let mut scratch = vec![vec![fuji_crate::FujiField::zero(); n]; n_rows];
    let mut sp = 0usize;

    fn dump(v: &[fuji_crate::FujiField], sp: usize) {
        let show = v.len().min(3);
        eprint!("[");
        for (i, f) in v.iter().take(show).enumerate() {
            if i > 0 { eprint!(", "); }
            let b = f.to_bytes();
            eprint!("0x{:02x}{:02x}{:02x}..", b[0], b[1], b[2]);
        }
        if v.len() > 3 { eprint!(", ...({})", v.len()); }
        eprintln!("]  (sp={})", sp);
    }

    fn dump_scalar(s: &fuji_crate::FujiField) {
        let b = s.to_bytes();
        eprint!("0x{:02x}{:02x}{:02x}..", b[0], b[1], b[2]);
    }

    for (ip, instr) in code.iter().enumerate() {
        eprint!("[{:4}] ", ip);
        match instr {
            fuji_crate::eval::Instr::LoadRotated { col, rotation } => {
                let p = &ctx.poly_table[*col as usize];
                let ds = p.domain_size;
                let row = &mut scratch[sp];
                for i in 0..n {
                    let idx = if *rotation >= 0 {
                        (ctx.chunk_start + i + *rotation as usize) % ds
                    } else {
                        let abs = (-rotation) as usize;
                        if ctx.chunk_start + i >= abs {
                            ctx.chunk_start + i - abs
                        } else {
                            ds - (abs - (ctx.chunk_start + i)) % ds
                        }
                    };
                    row[i] = p.data[idx % ds];
                }
                eprintln!("LOAD_ROTATED col={} rot={}", col, rotation);
                eprint!("       → "); dump(row, sp);
                sp += 1;
            }
            fuji_crate::eval::Instr::Add => {
                let a = sp - 2; let b = sp - 1;
                let tmp_a = scratch[a].clone();
                let tmp_b = scratch[b].clone();
                batch_field::add(&tmp_a, &tmp_b, &mut scratch[a], curve);
                eprintln!("ADD");
                eprint!("       → "); dump(&scratch[a], sp - 1);
                sp -= 1;
            }
            fuji_crate::eval::Instr::Sub => {
                let a = sp - 2; let b = sp - 1;
                let tmp_a = scratch[a].clone();
                let tmp_b = scratch[b].clone();
                batch_field::sub(&tmp_a, &tmp_b, &mut scratch[a], curve);
                eprintln!("SUB");
                eprint!("       → "); dump(&scratch[a], sp - 1);
                sp -= 1;
            }
            fuji_crate::eval::Instr::Mul => {
                let a = sp - 2; let b = sp - 1;
                let tmp_a = scratch[a].clone();
                let tmp_b = scratch[b].clone();
                batch_field::mul(&tmp_a, &tmp_b, &mut scratch[a], curve);
                eprintln!("MUL");
                eprint!("       → "); dump(&scratch[a], sp - 1);
                sp -= 1;
            }
            fuji_crate::eval::Instr::Scale { scalar_idx } => {
                let s = &ctx.scalars[*scalar_idx as usize];
                let tmp = scratch[sp - 1].clone();
                batch_field::scale(&tmp, s, &mut scratch[sp - 1], curve);
                eprint!("SCALE scalar_idx={} val=", scalar_idx);
                dump_scalar(s);
                eprintln!();
                eprint!("       → "); dump(&scratch[sp - 1], sp);
            }
            fuji_crate::eval::Instr::Fma { scalar_idx } => {
                let s = &ctx.scalars[*scalar_idx as usize];
                let a = sp - 2; let term = sp - 1;
                let tmp_a = scratch[a].clone();
                let tmp_term = scratch[term].clone();
                batch_field::scale(&tmp_a, s, &mut scratch[a], curve);
                let tmp_a2 = scratch[a].clone();
                batch_field::add(&tmp_a2, &tmp_term, &mut scratch[a], curve);
                eprint!("FMA scalar_idx={} val=", scalar_idx);
                dump_scalar(s);
                eprintln!();
                eprint!("       → "); dump(&scratch[a], sp - 1);
                sp -= 1;
            }
            fuji_crate::eval::Instr::LinearTerm { scalar_idx } => {
                let s = &ctx.scalars[*scalar_idx as usize];
                let row = &mut scratch[sp];
                let om = ctx.omega;
                // pow = omega^chunk_start
                let mut pow = om;
                for _ in 0..ctx.chunk_start {
                    batch_field::scale(&[pow], &om, &mut [pow], curve);
                }
                for i in 0..n {
                    if i == 0 {
                        batch_field::scale(&[*s], &pow, &mut [row[i]], curve);
                    } else {
                        batch_field::mul(&[row[i - 1]], &[om], &mut [row[i]], curve);
                    }
                }
                eprint!("LINEAR_TERM scalar_idx={} val=", scalar_idx);
                dump_scalar(s);
                eprintln!();
                eprint!("       → "); dump(row, sp);
                sp += 1;
            }
            fuji_crate::eval::Instr::Constant { scalar_idx } => {
                let s = &ctx.scalars[*scalar_idx as usize];
                scratch[sp].fill(*s);
                eprint!("CONSTANT scalar_idx={} val=", scalar_idx);
                dump_scalar(s);
                eprintln!();
                eprint!("       → "); dump(&scratch[sp], sp);
                sp += 1;
            }
            fuji_crate::eval::Instr::Negate => {
                let r = sp - 1;
                let zeros = vec![fuji_crate::FujiField::zero(); n];
                let tmp_r = scratch[r].clone();
                batch_field::sub(&zeros, &tmp_r, &mut scratch[r], curve);
                eprintln!("NEGATE");
                eprint!("       → "); dump(&scratch[r], sp);
            }
            fuji_crate::eval::Instr::CopyToOut => {
                eprintln!("COPY_TO_OUT  (out = scratch[0])");
                eprint!("       → "); dump(&scratch[0], 0);
            }
            fuji_crate::eval::Instr::Dup => {
                let src = scratch[sp - 1].clone();
                scratch[sp].copy_from_slice(&src);
                eprintln!("DUP");
                sp += 1;
            }
        }
    }
}

/// Struct representing the [`Ast::Mul`] case.
///
/// This struct exists to make the internals of this case private so that we don't
/// accidentally construct this case directly, because it can only be implemented for the
/// [`ExtendedLagrangeCoeff`] basis.
#[derive(Clone)]
pub(crate) struct AstMul<E, F: Field, B: Basis>(Arc<Ast<E, F, B>>, Arc<Ast<E, F, B>>);

impl<E, F: Field, B: Basis> fmt::Debug for AstMul<E, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("AstMul")
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

/// A polynomial operation backed by an [`Evaluator`].
#[derive(Clone)]
pub(crate) enum Ast<E, F: Field, B: Basis> {
    Poly(AstLeaf<E, B>),
    Add(Arc<Ast<E, F, B>>, Arc<Ast<E, F, B>>),
    Mul(AstMul<E, F, B>),
    Scale(Arc<Ast<E, F, B>>, F),
    /// Represents a linear combination of a vector of nodes and the powers of a
    /// field element, where the nodes are ordered from highest to lowest degree
    /// terms.
    DistributePowers(Arc<Vec<Ast<E, F, B>>>, F),
    /// The degree-1 term of a polynomial.
    ///
    /// The field element is the coefficient of the term in the standard basis, not the
    /// coefficient basis.
    LinearTerm(F),
    /// The degree-0 term of a polynomial.
    ///
    /// The field element is the same in both the standard and evaluation bases.
    ConstantTerm(F),
}

impl<E, F: Field, B: Basis> Ast<E, F, B> {
    pub fn distribute_powers<I: IntoIterator<Item = Self>>(i: I, base: F) -> Self {
        Ast::DistributePowers(Arc::new(i.into_iter().collect()), base)
    }

    /// Maximum recursion depth of the AST — used to size the scratch buffer stack.
    pub(crate) fn depth(&self) -> usize {
        match self {
            Ast::Poly(_) | Ast::LinearTerm(_) | Ast::ConstantTerm(_) => 0,
            Ast::Scale(a, _) => a.depth(),
            Ast::Add(a, b) | Ast::Mul(AstMul(a, b)) => 1 + a.depth().max(b.depth()),
            Ast::DistributePowers(terms, _) => terms.iter().map(|t| t.depth()).max().unwrap_or(0),
        }
    }
}

impl<E, F: Field, B: Basis> fmt::Debug for Ast<E, F, B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Poly(leaf) => f.debug_tuple("Poly").field(leaf).finish(),
            Self::Add(lhs, rhs) => f.debug_tuple("Add").field(lhs).field(rhs).finish(),
            Self::Mul(x) => f.debug_tuple("Mul").field(x).finish(),
            Self::Scale(base, scalar) => f.debug_tuple("Scale").field(base).field(scalar).finish(),
            Self::DistributePowers(terms, base) => f
                .debug_tuple("DistributePowers")
                .field(terms)
                .field(base)
                .finish(),
            Self::LinearTerm(x) => f.debug_tuple("LinearTerm").field(x).finish(),
            Self::ConstantTerm(x) => f.debug_tuple("ConstantTerm").field(x).finish(),
        }
    }
}

impl<E, F: Field, B: Basis> From<AstLeaf<E, B>> for Ast<E, F, B> {
    fn from(leaf: AstLeaf<E, B>) -> Self {
        Ast::Poly(leaf)
    }
}

impl<E, F: Field, B: Basis> Ast<E, F, B> {
    pub(crate) fn one() -> Self {
        Self::ConstantTerm(F::ONE)
    }
}

impl<E, F: Field, B: Basis> Neg for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        Ast::Scale(Arc::new(self), -F::ONE)
    }
}

impl<E: Clone, F: Field, B: Basis> Neg for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn neg(self) -> Self::Output {
        -(self.clone())
    }
}

impl<E, F: Field, B: Basis> Add for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: Self) -> Self::Output {
        Ast::Add(Arc::new(self), Arc::new(other))
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Add<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self.clone() + other.clone()
    }
}

impl<E, F: Field, B: Basis> Add<AstLeaf<E, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn add(self, other: AstLeaf<E, B>) -> Self::Output {
        Ast::Add(Arc::new(self), Arc::new(other.into()))
    }
}

impl<E, F: Field, B: Basis> Sub for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: Self) -> Self::Output {
        self + (-other)
    }
}

impl<'a, E: Clone, F: Field, B: Basis> Sub<&'a Ast<E, F, B>> for &'a Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: &'a Ast<E, F, B>) -> Self::Output {
        self + &(-other)
    }
}

impl<E, F: Field, B: Basis> Sub<AstLeaf<E, B>> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn sub(self, other: AstLeaf<E, B>) -> Self::Output {
        self + (-Ast::from(other))
    }
}

impl<E, F: Field> Mul for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: Self) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, LagrangeCoeff>> for &'a Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, LagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<AstLeaf<E, LagrangeCoeff>> for Ast<E, F, LagrangeCoeff> {
    type Output = Ast<E, F, LagrangeCoeff>;

    fn mul(self, other: AstLeaf<E, LagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other.into())))
    }
}

impl<E, F: Field> Mul for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: Self) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other)))
    }
}

impl<'a, E: Clone, F: Field> Mul<&'a Ast<E, F, ExtendedLagrangeCoeff>>
    for &'a Ast<E, F, ExtendedLagrangeCoeff>
{
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: &'a Ast<E, F, ExtendedLagrangeCoeff>) -> Self::Output {
        self.clone() * other.clone()
    }
}

impl<E, F: Field> Mul<AstLeaf<E, ExtendedLagrangeCoeff>> for Ast<E, F, ExtendedLagrangeCoeff> {
    type Output = Ast<E, F, ExtendedLagrangeCoeff>;

    fn mul(self, other: AstLeaf<E, ExtendedLagrangeCoeff>) -> Self::Output {
        Ast::Mul(AstMul(Arc::new(self), Arc::new(other.into())))
    }
}

impl<E, F: Field, B: Basis> Mul<F> for Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Arc::new(self), other)
    }
}

impl<E: Clone, F: Field, B: Basis> Mul<F> for &Ast<E, F, B> {
    type Output = Ast<E, F, B>;

    fn mul(self, other: F) -> Self::Output {
        Ast::Scale(Arc::new(self.clone()), other)
    }
}

impl<E: Clone, F: Field> MulAssign for Ast<E, F, ExtendedLagrangeCoeff> {
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs)
    }
}

/// Operations which can be performed over a given basis.
pub(crate) trait BasisOps: Basis {
    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self>;
    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F>;
    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F>;
    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F>;

    // ── _into variants (avoid Vec allocation) ─────────────────────
    fn constant_term_into<F: Field>(out: &mut [F], scalar: F);
    fn linear_term_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    );
    fn get_chunk_of_rotated_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    );
}

impl BasisOps for Coeff {
    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self> {
        domain.empty_coeff()
    }

    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let mut chunk = vec![F::ZERO; cmp::min(chunk_size, poly_len - chunk_size * chunk_index)];
        if chunk_index == 0 {
            chunk[0] = scalar;
        }
        chunk
    }

    fn constant_term_into<F: Field>(out: &mut [F], scalar: F) {
        out.fill(F::ZERO);
        out[0] = scalar;
    }

    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        _: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let mut chunk = vec![F::ZERO; cmp::min(chunk_size, poly_len - chunk_size * chunk_index)];
        if chunk_size == 1 && chunk_index == 1 {
            chunk[0] = scalar;
        } else if chunk_index == 0 {
            chunk[1] = scalar;
        }
        chunk
    }

    fn linear_term_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        _domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) {
        out.fill(F::ZERO);
        if chunk_size == 1 && chunk_index == 1 && !out.is_empty() {
            out[0] = scalar;
        } else if chunk_index == 0 && out.len() > 1 {
            out[1] = scalar;
        }
    }

    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        _: &EvaluationDomain<F>,
        _: usize,
        _: usize,
        _: &Polynomial<F, Self>,
        _: Rotation,
    ) -> Vec<F> {
        panic!("Can't rotate polynomials in the standard basis")
    }

    fn get_chunk_of_rotated_into<F: WithSmallOrderMulGroup<3>>(
        _out: &mut [F],
        _domain: &EvaluationDomain<F>,
        _chunk_size: usize,
        _chunk_index: usize,
        _poly: &Polynomial<F, Self>,
        _rotation: Rotation,
    ) {
        panic!("Can't rotate polynomials in the standard basis")
    }
}

impl BasisOps for LagrangeCoeff {
    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self> {
        domain.empty_lagrange()
    }

    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        vec![scalar; cmp::min(chunk_size, poly_len - chunk_size * chunk_index)]
    }

    fn constant_term_into<F: Field>(out: &mut [F], scalar: F) {
        out.fill(scalar);
    }

    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        let omega = domain.get_omega();
        let start = chunk_size * chunk_index;
        (0..cmp::min(chunk_size, poly_len - start))
            .scan(omega.pow_vartime([start as u64]) * scalar, |acc, _| {
                let ret = *acc;
                *acc *= omega;
                Some(ret)
            })
            .collect()
    }

    fn linear_term_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) {
        let omega = domain.get_omega();
        let start = chunk_size * chunk_index;
        for (i, elem) in out.iter_mut().enumerate() {
            *elem = omega.pow_vartime([(start + i) as u64]) * scalar;
        }
    }

    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        _: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F> {
        poly.get_chunk_of_rotated(rotation, chunk_size, chunk_index)
    }

    fn get_chunk_of_rotated_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        _domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) {
        poly.get_chunk_of_rotated_helper_into(
            out,
            rotation.0 < 0,
            rotation.0.unsigned_abs() as usize,
            chunk_size,
            chunk_index,
        );
    }
}

impl BasisOps for ExtendedLagrangeCoeff {
    fn empty_poly<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
    ) -> Polynomial<F, Self> {
        domain.empty_extended()
    }

    fn constant_term<F: Field>(
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        vec![scalar; cmp::min(chunk_size, poly_len - chunk_size * chunk_index)]
    }

    fn constant_term_into<F: Field>(out: &mut [F], scalar: F) {
        out.fill(scalar);
    }

    fn linear_term<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        poly_len: usize,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) -> Vec<F> {
        // Take every power of the extended omega within the chunk, and multiply by scalar.
        let omega = domain.get_extended_omega();
        let start = chunk_size * chunk_index;
        (0..cmp::min(chunk_size, poly_len - start))
            .scan(
                omega.pow_vartime([start as u64]) * F::ZETA * scalar,
                |acc, _| {
                    let ret = *acc;
                    *acc *= omega;
                    Some(ret)
                },
            )
            .collect()
    }

    fn linear_term_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        scalar: F,
    ) {
        let omega = domain.get_extended_omega();
        let start = chunk_size * chunk_index;
        for (i, elem) in out.iter_mut().enumerate() {
            *elem = omega.pow_vartime([(start + i) as u64]) * F::ZETA * scalar;
        }
    }

    fn get_chunk_of_rotated<F: WithSmallOrderMulGroup<3>>(
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) -> Vec<F> {
        domain.get_chunk_of_rotated_extended(poly, rotation, chunk_size, chunk_index)
    }

    fn get_chunk_of_rotated_into<F: WithSmallOrderMulGroup<3>>(
        out: &mut [F],
        domain: &EvaluationDomain<F>,
        chunk_size: usize,
        chunk_index: usize,
        poly: &Polynomial<F, Self>,
        rotation: Rotation,
    ) {
        domain.get_chunk_of_rotated_extended_into(out, poly, rotation, chunk_size, chunk_index);
    }
}

#[cfg(test)]
mod tests {
    use group::ff::Field;
    use pasta_curves::pallas;

    use super::{get_chunk_params, new_evaluator, Ast, BasisOps, Evaluator};
    use crate::poly::{Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff};

    #[test]
    fn short_chunk_regression_test() {
        // Pick the smallest polynomial length that is guaranteed to produce a short chunk
        // on this machine.
        let k = match (1..16)
            .map(|k| (k, get_chunk_params(1 << k)))
            .find(|(k, (chunk_size, num_chunks))| (1 << k) < chunk_size * num_chunks)
            .map(|(k, _)| k)
        {
            Some(k) => k,
            None => {
                // We are on a machine with a power-of-two number of threads, and cannot
                // trigger the bug.
                eprintln!(
                    "can't find a polynomial length for short_chunk_regression_test; skipping"
                );
                return;
            }
        };
        eprintln!("Testing short-chunk regression with k = {}", k);

        fn test_case<E: Copy + Send + Sync, B: BasisOps>(
            k: u32,
            mut evaluator: Evaluator<E, pallas::Base, B>,
        ) {
            // Instantiate the evaluator with a trivial polynomial.
            let domain = EvaluationDomain::new(1, k);
            evaluator.register_poly(B::empty_poly(&domain));

            // With the bug present, these will panic.
            let _ = evaluator.evaluate(&Ast::ConstantTerm(pallas::Base::ZERO), &domain);
            let _ = evaluator.evaluate(&Ast::LinearTerm(pallas::Base::ZERO), &domain);
        }

        test_case(k, new_evaluator::<_, _, Coeff>(|| {}));
        test_case(k, new_evaluator::<_, _, LagrangeCoeff>(|| {}));
        test_case(k, new_evaluator::<_, _, ExtendedLagrangeCoeff>(|| {}));
    }
}
