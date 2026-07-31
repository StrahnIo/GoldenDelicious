use std::process::Command;
fn main() {
    // Time just spawning and waiting for 'true'
    let t0 = std::time::Instant::now();
    for _ in 0..10 {
        Command::new("true").spawn().unwrap().wait().unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 10.0;
    println!("fork+exec 'true': {:.3} ms avg", ms);
    
    // Time spawning fuji_eval with a no-op (no scalars file = exits immediately)
    let eval = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuji_eval");
    let prove = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_tmp/prove_1785179193979679000.fuji");
    
    let t0 = std::time::Instant::now();
    for _ in 0..3 {
        let mut c = Command::new(eval.to_str().unwrap())
            .arg(prove.to_str().unwrap())
            .arg("/nonexistent")
            .spawn().unwrap();
        c.wait().unwrap();
    }
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / 3.0;
    println!("fork+exec fuji_eval (fail): {:.3} ms avg", ms);
}
