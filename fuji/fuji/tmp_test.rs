fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = std::fs::read(&args[1]).unwrap();
    let mut bc_ptr = std::ptr::null_mut();
    unsafe { fuji_sys::fuji_bytecode_read(data.as_ptr(), data.len(), &mut bc_ptr); }
    let bc = unsafe { &*bc_ptr };
    let fn_ptr: fuji_sys::jit_chunk_fn = unsafe { fuji_sys::fuji_jit_compile_from_bc(bc_ptr) };
    println!("fn_ptr in Rust: {:p}", fn_ptr);
    unsafe { fuji_sys::fuji_jit_free(fn_ptr); }
    unsafe { fuji_sys::fuji_bytecode_free(bc_ptr); }
}
