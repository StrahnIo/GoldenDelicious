use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let mut bc_ptr = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };

    let cl = bc.chunk_len as usize;
    let md = bc.max_depth as usize;
    let num_chunks = (bc.n_instr as usize + cl - 1) / cl;

    let fn_ptr: fuji_sys::jit_chunk_fn = unsafe {
        fuji_sys::fuji_jit_compile(
            bc.code, bc.n_instr, bc.poly_table, bc.n_polys,
            bc.chunk_len, bc.max_depth, bc.curve,
        )
    };
    let code_size = unsafe { fuji_sys::fuji_jit_code_size() };
    let p = unsafe { fuji_sys::fuji_field_modulus(bc.curve) as *const std::ffi::c_void };
    let pinv: u64 = match bc.curve { 0 => 0x992d30ecffffffff, _ => 0x8c46eb20ffffffff };

    // Pre-allocate per-chunk buffers
    let mut scs: Vec<Vec<fuji_sys::fuji_field>> = (0..num_chunks)
        .map(|_| vec![fuji_sys::fuji_field { limbs: [0u8; 32] }; md * cl]).collect();
    let mut outs: Vec<Vec<fuji_sys::fuji_field>> = (0..num_chunks)
        .map(|_| vec![fuji_sys::fuji_field { limbs: [0u8; 32] }; cl]).collect();

    // True sequential: all 41 chunks in a loop (like C true_seq test)
    let t0 = Instant::now();
    for ch in 0..num_chunks {
        let chunk_start = (ch * cl) as i32;
        unsafe {
            (fn_ptr)(
                chunk_start,
                scs[ch].as_mut_ptr(),
                outs[ch].as_mut_ptr(),
                bc.poly_table,
                bc.scalars,
                bc.omega,
                p,
                pinv,
            );
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("Rust true seq: {:.2} ms ({:.2} ms/chunk)", ms, ms / num_chunks as f64);

    unsafe { fuji_sys::fuji_jit_free(fn_ptr); }
    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
