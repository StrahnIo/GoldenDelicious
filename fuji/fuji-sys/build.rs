fn main() {
    let lib_dir = std::env::var("FUJI_LIB_DIR").unwrap_or_else(|_| {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap().parent().unwrap()
            .to_string_lossy().to_string()
    });

    println!("cargo:rustc-link-search={}", lib_dir);

    let static_path = std::path::Path::new(&lib_dir).join("libfuji.a");
    if static_path.exists() {
        println!("cargo:rustc-link-lib=static=fuji");
    } else {
        println!("cargo:rustc-link-lib=dylib=fuji");
    }

    println!("cargo:rerun-if-changed={}", static_path.display());
    println!("cargo:rerun-if-env-changed=FUJI_LIB_DIR");
}
