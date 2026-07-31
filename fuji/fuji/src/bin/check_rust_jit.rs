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
    let fn_addr = fn_ptr as *const u32;
    let first = unsafe { *fn_addr };
    println!("fn_ptr={:p} first_inst=0x{:08x}", fn_ptr, first);
    println!("(should be 0xd10183ff = sub sp, sp, #96)");
    println!("Match: {}", first == 0xd10183ff);

    let code_size = unsafe { fuji_sys::fuji_jit_code_size() };
    println!("code_size={}", code_size);

    // Print instructions around fn_ptr
    print!("Next:");
    for i in 0..5 {
        print!(" 0x{:08x}", unsafe { *((fn_addr as usize + i*4) as *const u32) });
    }
    println!();

    unsafe { fuji_sys::fuji_jit_free(fn_ptr); }
    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
