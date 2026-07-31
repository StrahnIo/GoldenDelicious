// Benchmark using fuji_eval binary IPC
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 { eprintln!("Usage: bench_jit_par <file.fuji>"); return; }

    let data = std::fs::read(&args[1]).expect("read file");
    let mut bc_ptr: *mut fuji_sys::fuji_bytecode = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };
    let num_chunks = ((bc.n_instr + bc.chunk_len - 1) / bc.chunk_len) as usize;
    let n = num_chunks * bc.chunk_len as usize;
    let n_scalars = bc.n_scalars as usize;

    println!("File: {}  ({} chunks, {} scalars)", args[1], num_chunks, n_scalars);

    // Copy scalars as raw bytes
    let scalars_bytes = unsafe {
        std::slice::from_raw_parts(bc.scalars as *const u8, n_scalars * 32)
    }.to_vec();

    // Spawn the thin C binary (at project root, relative to crate manifest)
    let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let eval_path = crate_dir.join("../../fuji_eval");
    let eval_path = std::fs::canonicalize(&eval_path).unwrap_or_else(|_|
        crate_dir.join("fuji_eval"));
    let prove_path = std::fs::canonicalize(&args[1]).unwrap();
    let mut eval = fuji::eval::EvalChild::spawn(
        eval_path.to_str().unwrap(),
        prove_path.to_str().unwrap(),
        n_scalars, n);

    let mut results = vec![0u8; n * 32];

    // First prove
    let t0 = std::time::Instant::now();
    match eval.eval(&scalars_bytes, &mut results) {
        Ok(()) => {
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            println!("  par: {:7.2} ms ({:.2} ms/chunk)", ms, ms / num_chunks as f64);
            println!("  first byte: {:02x}", results[0]);
        }
        Err(e) => {
            println!("  ERROR: {}", e);
        }
    }

    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
