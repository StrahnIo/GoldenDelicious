use std::io::{Read, Write};
use std::process::{Command, Stdio};
fn main() {
    let eval = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fuji_eval");
    let prove = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test_tmp/prove_1785179193979679000.fuji");
    
    let mut child = Command::new(eval.to_str().unwrap())
        .arg(prove.to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null()) // ignore stderr for now
        .spawn()
        .expect("spawn");
    
    let mut stdin = child.stdin.take().unwrap();
    
    // Try to write one byte first (instead of full block)
    let result = stdin.write(&[0u8]);
    eprintln!("write one byte: {:?}", result);
    
    let scalars = vec![0u8; 5711 * 32];
    eprintln!("Writing {} bytes...", scalars.len());
    stdin.write_all(&scalars).expect("write_all");
    eprintln!("Write done");
    drop(stdin);
    
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
