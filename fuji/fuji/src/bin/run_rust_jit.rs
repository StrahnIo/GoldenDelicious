// Minimal Rust test: load .fuji, compile JIT, run chunk 0, time it
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let mut bc_ptr = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };

    let fn_ptr: fuji_sys::jit_chunk_fn = unsafe {
        fuji_sys::fuji_jit_compile(
            bc.code, bc.n_instr, bc.poly_table, bc.n_polys,
            bc.chunk_len, bc.max_depth, bc.curve,
        )
    };
    println!("fn_ptr={:p} code_size={}", fn_ptr, unsafe { fuji_sys::fuji_jit_code_size() });

    let cl = bc.chunk_len as usize;
    let md = bc.max_depth as usize;
    let mut scratch = vec![fuji_sys::fuji_field { limbs: [0u8; 32] }; md * cl];
    let mut out = vec![fuji_sys::fuji_field { limbs: [0u8; 32] }; cl];
    let p = unsafe { fuji_sys::fuji_field_modulus(bc.curve) as *const std::ffi::c_void };
    let pinv: u64 = match bc.curve { 0 => 0x992d30ecffffffff, _ => 0x8c46eb20ffffffff };

    // Time 10 iterations
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        scratch.fill(fuji_sys::fuji_field { limbs: [0u8; 32] });
        out.fill(fuji_sys::fuji_field { limbs: [0u8; 32] });
        unsafe {
            (fn_ptr)(
                0, scratch.as_mut_ptr(), out.as_mut_ptr(),
                bc.poly_table, bc.scalars, bc.omega, p, pinv,
            );
        }
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("Per chunk: {:.3} ms", ms);
    println!("First byte: {:02x}", out[0].limbs[0]);

    // Compare with C interpreter
    let mut c_out = vec![fuji_sys::fuji_field { limbs: [0u8; 32] }; cl];
    unsafe {
        fuji_sys::fuji_eval_execute(
            bc.code, bc.n_instr, bc.poly_table, bc.n_polys,
            bc.scalars, bc.n_scalars,
            scratch.as_mut_ptr(), md as i32,
            0, cl as i32, bc.omega, bc.curve, c_out.as_mut_ptr(),
            std::ptr::null_mut(),
        );
    }
    println!("C first byte: {:02x}", c_out[0].limbs[0]);

    unsafe { fuji_sys::fuji_jit_free(fn_ptr); }
    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
