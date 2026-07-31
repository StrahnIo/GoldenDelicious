// Direct FFI call to fuji_jit_bench_exec — minimal Rust overhead
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let mut bc_ptr = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };

    let fn_ptr: fuji_sys::jit_chunk_fn = unsafe { fuji_sys::fuji_jit_compile_from_bc(bc_ptr) };
    let mut n_scalars: i32 = 0;
    let scalars = unsafe { fuji_sys::fuji_jit_bytecode_scalars(bc_ptr, &mut n_scalars) };

    let cl = bc.chunk_len;
    let md = bc.max_depth;
    let num_chunks = (bc.n_instr + cl - 1) / cl;
    let n = num_chunks * cl;

    // Warmup call
    unsafe {
        fuji_sys::fuji_jit_bench_exec(
            fn_ptr, bc.code, bc.n_instr, bc.poly_table, bc.n_polys,
            scalars, n_scalars, bc.omega, bc.curve,
            cl, md, n, 0,
        );
    }

    // Timed call — sequential
    let t0 = std::time::Instant::now();
    let r = unsafe {
        fuji_sys::fuji_jit_bench_exec(
            fn_ptr, bc.code, bc.n_instr, bc.poly_table, bc.n_polys,
            scalars, n_scalars, bc.omega, bc.curve,
            cl, md, n, 0,
        )
    };
    let seq_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("seq via ffi: {:.3} ms ({:.3} ms/chunk) match={}",
             seq_ms, seq_ms / r.n_chunks as f64, r.match_);

    // Timed call — dispatch_apply
    let t0 = std::time::Instant::now();
    let r = unsafe {
        fuji_sys::fuji_jit_bench_exec(
            fn_ptr, bc.code, bc.n_instr, bc.poly_table, bc.n_polys,
            scalars, n_scalars, bc.omega, bc.curve,
            cl, md, n, 2,
        )
    };
    let par_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("par via ffi: {:.3} ms ({:.3} ms/chunk) match={}",
             par_ms, par_ms / r.n_chunks as f64, r.match_);

    unsafe { fuji_sys::fuji_jit_free(fn_ptr); }
    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
