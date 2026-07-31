fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let mut bc_ptr = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };

    let cl = bc.chunk_len;
    let md = bc.max_depth;
    let ni = bc.n_instr;
    let num_chunks = (ni + cl - 1) / cl;
    let n = num_chunks * cl;

    println!("cl={} md={} ni={} num={} n={}", cl, md, ni, num_chunks, n);
    println!("poly_table={:p} scalars={:p} omega={:p} curve={}",
             bc.poly_table, bc.scalars, bc.omega, bc.curve);
    println!("n_polys={} n_scalars={}", bc.n_polys, bc.n_scalars);

    let res = unsafe {
        fuji_sys::fuji_jit_bench(
            bc.code, ni,
            bc.poly_table, bc.n_polys,
            bc.scalars, bc.n_scalars,
            bc.omega,
            bc.curve,
            cl, md,
            n,
            0, // seq
        )
    };
    println!("Seq: {:.3} ms ({:.3} ms/chunk) match={} n_chunks={}",
             res.exec_ms, res.exec_ms / res.n_chunks as f64, res.match_, res.n_chunks);

    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
