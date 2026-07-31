use std::io::Read;
use std::process::{Command, Stdio};
fn main() {
    let eval = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuji_eval");
    let prove = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_tmp/prove_1785179193979679000.fuji");
    
    let mut child = Command::new(eval.to_str().unwrap())
        .arg(prove.to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    
    // Read stderr first to see RDY
    let mut stderr = child.stderr.take().unwrap();
    let mut err_buf = [0u8; 8];
    let n = stderr.read(&mut err_buf).unwrap_or(0);
    eprintln!("stderr first bytes ({}): {:?}", n, &err_buf[..n]);
    
    // Now try writing to stdin
    let mut stdin = child.stdin.take().unwrap();
    use std::io::Write;
    let scalars = vec![0u8; 5711 * 32];
    stdin.write_all(&scalars).expect("write");
    drop(stdin);
    
    // Read stdout
    let mut stdout = child.stdout.take().unwrap();
    let mut buf = vec![0u8; 41 * 410 * 32];
    let mut off = 0;
    while off < buf.len() {
        let n = stdout.read(&mut buf[off..]).unwrap_or(0);
        if n == 0 { break; }
        off += n;
    }
    eprintln!("stdout: read {} bytes", off);
    println!("DONE first={:02x}", buf[0]);
}
