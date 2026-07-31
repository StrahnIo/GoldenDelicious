use std::io::{Read, Write};
use std::process::{Command, Stdio};
fn main() {
    let anchor = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let eval = anchor.join("../../fuji_eval");
    let prove = anchor.join("../../test_tmp/prove_1785179193979679000.fuji");
    eprintln!("eval: {:?} exists={}", eval, eval.exists());
    eprintln!("prove: {:?} exists={}", prove, prove.exists());

    let mut child = Command::new(eval.to_str().unwrap())
        .arg(prove.to_str().unwrap())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn");
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    let scalars = vec![0u8; 5711 * 32];
    eprintln!("Writing...");
    stdin.write_all(&scalars).expect("write_all");
    std::mem::drop(stdin); // close stdin so child knows EOF

    eprintln!("Reading...");
    let mut buf = vec![0u8; 41 * 410 * 32];
    let mut off = 0;
    while off < buf.len() {
        let n = stdout.read(&mut buf[off..]).unwrap();
        if n == 0 { break; }
        off += n;
    }
    eprintln!("Read {} bytes", off);
    let mut err = String::new();
    stderr.read_to_string(&mut err).unwrap();
    if !err.is_empty() { eprintln!("child stderr: {}", err); }
    println!("DONE first={:02x}", buf[0]);
}
