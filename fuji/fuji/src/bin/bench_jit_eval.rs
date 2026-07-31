// Benchmark JitEval from Rust — in-process JIT, no subprocess.
use fuji::eval::JitEval;
use fuji::FujiField;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("Usage: bench_jit_eval <file.fuji> [scalars.bin]"); return; }

    let bc_buf = std::fs::read(&args[1]).expect("read .fuji file");
    let mut bc_ptr: *mut fuji_sys::fuji_bytecode = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(bc_buf.as_ptr(), bc_buf.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };
    let n_scalars = bc.n_scalars as usize;
    let cl = bc.chunk_len as usize;
    let num_chunks = ((bc.n_instr as usize) + cl - 1) / cl;
    let n_results = num_chunks * cl;

    // Read scalars
    let scalars: Vec<FujiField>;
    if args.len() >= 3 {
        let raw = std::fs::read(&args[2]).expect("read scalars file");
        assert_eq!(raw.len(), n_scalars * 32);
        scalars = unsafe { std::mem::transmute::<Vec<u8>, Vec<FujiField>>(raw) };
    } else {
        let mut n = 0i32;
        let ptr = unsafe { fuji_sys::fuji_jit_bytecode_scalars(bc_ptr, &mut n) };
        scalars = unsafe { std::slice::from_raw_parts(ptr as *const FujiField, n as usize).to_vec() };
    }

    let mut results = vec![FujiField::zero(); n_results];

    // Compile once
    let t0 = std::time::Instant::now();
    let jit = JitEval::compile(&bc_buf, n_scalars, n_results).expect("compile");
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Warmup eval
    jit.eval(&scalars, &mut results).expect("warmup");

    // Timed evals
    let n_runs = 5;
    let mut times = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        results.fill(FujiField::zero());
        let t0 = std::time::Instant::now();
        jit.eval(&scalars, &mut results).expect("eval");
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let min_ms = times.iter().cloned().fold(f64::MAX, f64::min);
    let max_ms = times.iter().cloned().fold(f64::MIN, f64::max);
    let avg_ms = times.iter().sum::<f64>() / times.len() as f64;

    println!("File: {}  ({} instr, {} chunks, {}/cl)", args[1], bc.n_instr, num_chunks, cl);
    println!("Compile: {:.2} ms", compile_ms);
    println!("JitEval (Rust, {} runs):", n_runs);
    println!("  min: {:7.2} ms", min_ms);
    println!("  avg: {:7.2} ms", avg_ms);
    println!("  max: {:7.2} ms", max_ms);
    println!("  first byte: {:02x}", results[0].0.limbs[0]);

    // jit dropped → cache freed
    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
