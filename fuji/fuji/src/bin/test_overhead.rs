// Measure just the spawn + file write overhead (no actual work)
use std::io::Read;
use std::process::{Command, Stdio};

fn main() {
    let eval = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuji_eval");
    let prove = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_tmp/prove_1785179193979679000.fuji");

    // Create temp scalars file
    let scalars = vec![0u8; 5711 * 32];
    std::fs::write("/tmp/overhead_test.bin", &scalars).unwrap();

    // Time just the spawn + file delete
    let t0 = std::time::Instant::now();
    for _ in 0..5 {
        let _ = std::fs::write("/tmp/overhead_test.bin", &scalars);
        let mut child = Command::new(eval.to_str().unwrap())
            .arg(prove.to_str().unwrap())
            .arg("/tmp/overhead_test.bin")
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn().unwrap();
        let mut out = Vec::new();
        child.stdout.take().unwrap().read_to_end(&mut out).unwrap();
        child.wait().unwrap();
        let _ = std::fs::remove_file("/tmp/overhead_test.bin");
    }
    let avg = t0.elapsed().as_secs_f64() / 5.0 * 1000.0;
    println!("Avg per prove (spawn + file + run + read): {:.1} ms", avg);
}
