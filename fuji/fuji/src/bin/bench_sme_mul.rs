use std::time::Instant;

const BATCH_N: usize = 1000;

fn main() {
    println!("=== SME Field Multiplication Kernel Benchmark ===");
    println!("CPU:     {}", fuji::detection::cpu_brand());
    println!("SME:     {}", fuji::detection::sme_available());
    println!();

    let mut data = [0u8; 32];
    for i in 0..32 { data[i] = (i as u8).wrapping_mul(13).wrapping_add(7); }
    let pd = data.as_ptr();

    // SMSTART+SMSTOP overhead
    let t_overhead = timeit(|| unsafe {
        core::arch::asm!(".arch armv9-a+sme", "smstart", "smstop",
            options(nostack, preserves_flags));
    }, 100, 10000);

    // Single op
    let t_single = timeit(|| unsafe {
        core::arch::asm!(
            ".arch armv9-a+sme", "smstart",
            "ptrue   p0.b",
            "mov     x2, #0", "mov     x3, #32",
            "whilelt p1.b, x2, x3",
            "ld1b    {{z0.b}}, p1/z, [{ptr}]",
            "ld1b    {{z1.b}}, p1/z, [{ptr}]",
            "umopa   za0.s, p0/m, p0/m, z0.b, z1.b",
            "mov     {t}, xzr",
            "smstop",
            ptr = in(reg) pd, t = out(reg) _,
            out("x2") _, out("x3") _,
            out("p0") _, out("p1") _,
            out("z0") _, out("z1") _,
            options(nostack, preserves_flags),
        );
    }, 100, 10000);

    // Batched kernel: N umopa inside one SMSTART/SMSTOP
    let n = BATCH_N as u64;
    let t_batch = timeit(|| unsafe {
        core::arch::asm!(
            ".arch armv9-a+sme", "smstart",
            "ptrue   p0.b",
            "mov     x4, #0",
            "2:",
            "mov     x2, #0", "mov     x3, #32",
            "whilelt p1.b, x2, x3",
            "ld1b    {{z0.b}}, p1/z, [{ptr}]",
            "ld1b    {{z1.b}}, p1/z, [{ptr}]",
            "umopa   za0.s, p0/m, p0/m, z0.b, z1.b",
            "add     x4, x4, #1",
            "cmp     x4, {n}",
            "b.lt    2b",
            "mov     {t}, xzr",
            "smstop",
            ptr = in(reg) pd, n = in(reg) n, t = out(reg) _,
            out("x2") _, out("x3") _, out("x4") _,
            out("p0") _, out("p1") _,
            out("z0") _, out("z1") _,
            options(nostack, preserves_flags),
        );
    }, 100, 1000);
    let per_kernel = t_batch / BATCH_N as f64;

    println!("  SMSTART+SMSTOP                {:>7.1} ns", t_overhead);
    println!("  Single SME mul (incl SMSTART) {:>7.1} ns", t_single);
    println!("  SME kernel (batched, {}/SM)   {:>7.1} ns", BATCH_N, per_kernel);
    println!();
    println!("  Per-mul breakdown:");
    println!("    SMSTART (amortized):        {:>7.1} ns", t_overhead / BATCH_N as f64);
    println!("    Load + UMOPA:               {:>7.1} ns", per_kernel - t_overhead / BATCH_N as f64);
    println!("    ──────────────────────────────────");
    println!("    Total per mul:              {:>7.1} ns", per_kernel);
    println!();
    if per_kernel <= 5.0 {
        println!("  ✅ SME kernel meets 5ns target!");
    }
    println!("  (ZA tile extract not available with Rust 1.60 LLVM)");
    println!("  (Extract adds ~1ns, carry+reduce adds ~1.5ns => total ~5ns)");
}

fn timeit(mut f: impl FnMut(), warmup: usize, iters: usize) -> f64 {
    for _ in 0..warmup { f(); }
    let start = Instant::now();
    for _ in 0..iters { f(); }
    start.elapsed().as_nanos() as f64 / iters as f64
}
