use std::io::{Read, Write};
use std::process::{Command, Stdio};
fn main() {
    let mut child = Command::new("/tmp/test_pipe")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn().unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut stdout = child.stdout.take().unwrap();
    let mut stderr = child.stderr.take().unwrap();

    stdin.write_all(b"hello").expect("write");
    drop(stdin);
    let mut out = String::new();
    stdout.read_to_string(&mut out).unwrap();
    let mut err = String::new();
    stderr.read_to_string(&mut err).unwrap();
    eprintln!("stderr: {}", err.trim());
    println!("stdout: {:?}", out);
}
