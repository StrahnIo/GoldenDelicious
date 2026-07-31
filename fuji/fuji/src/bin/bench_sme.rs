use std::time::{Duration, Instant};

fn black_box<T>(x: T) -> T {
    unsafe { std::ptr::read_volatile(&x) }
}

const WARMUP: usize = 2000;
const ITERATIONS: usize = 100000;

fn main() {
    println!("=== Fuji Field Arithmetic Benchmarks ===");
    println!("CPU:     {}", fuji::detection::cpu_brand());
    println!("SME:     {}", fuji::detection::sme_available());
    println!("AMX:     {}", fuji::detection::amx_available());
    println!();

    let args: Vec<String> = std::env::args().collect();
    let all = args.len() == 1;
    let has = |s: &str| args.iter().any(|x| x == s);

    if all {
        let oh = bench_overhead();
        let st = bench_setup();
        let tot = bench_umopa();
        let ax = bench_amx();
        let mg = bench_montgomery();
        let oh_raw = oh.1;
        let st_raw = st.1;
        let tot_raw = tot.1;

        let ptrue_ns = if st_raw > oh_raw {
            (st_raw - oh_raw) as f64 / ITERATIONS as f64
        } else {
            0.0
        };
        let umopa_only = if tot_raw > st_raw {
            (tot_raw - st_raw) as f64 / ITERATIONS as f64
        } else {
            0.0
        };
        let raw_ns = (tot_raw - oh_raw) as f64 / ITERATIONS as f64;

        println!();
        println!("  Breakdown (integer arithmetic):");
        println!("    SMSTART+SMSTOP           {:>9.1} ns", oh.0);
        println!("    PTRUE+LD1B               {:>9.1} ns  (setup - overhead)", ptrue_ns);
        println!("    Raw UMOPA                {:>9.1} ns  (total - setup)", umopa_only);
        println!("    Raw UMOPA (alt)          {:>9.1} ns  (total - overhead)", raw_ns);
        println!();
        println!("  Comparison:");
        let ratio = if raw_ns > 0.0 { ax.0 / raw_ns } else { 0.0 };
        println!(
            "    AMX field_mul           {:>9.1} ns/op ({}× slower than UMOPA)",
            ax.0,
            ratio.round() as u64
        );
        let ratio_mg = if raw_ns > 0.0 { mg.0 / raw_ns } else { 0.0 };
        println!(
            "    Montgomery field_mul    {:>9.1} ns/op ({}× slower than UMOPA)",
            mg.0,
            ratio_mg.round() as u64
        );
    } else {
        if has("--overhead") { bench_overhead(); }
        if has("--setup") { bench_setup(); }
        if has("--umopa") { bench_umopa(); }
        if has("--amx") { bench_amx(); }
        if has("--montgomery") { bench_montgomery(); }
    }
}

struct BenchResult(f64, i128);

fn bench_overhead() -> BenchResult {
    let (dur, _) = time_it(|| {
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sme",
                "smstart",
                "smstop",
                out("x0") _,
                out("x1") _,
                options(nostack, preserves_flags),
            );
        }
    });
    let ns = dur.as_nanos() as i128 as f64 / ITERATIONS as f64;
    println!("  SMSTART+SMSTOP           {:>9.1} ns/op", ns);
    BenchResult(ns, dur.as_nanos() as i128)
}

fn bench_setup() -> BenchResult {
    let mut a = [0u8; 32];
    for i in 0..32 {
        a[i] = (i as u8).wrapping_mul(7);
    }
    let pa = &a as *const u8;

    let (dur, _) = time_it(|| {
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sme",
                "smstart",
                "ptrue   p0.b",
                "mov     x2, #0",
                "mov     x3, #32",
                "whilelt p1.b, x2, x3",
                "ld1b    {{z0.b}}, p1/z, [{a}]",
                "smstop",
                a = in(reg) pa,
                out("x2") _,
                out("x3") _,
                out("p0") _,
                out("p1") _,
                out("z0") _,
                options(nostack, preserves_flags),
            );
        }
    });
    let ns = dur.as_nanos() as i128 as f64 / ITERATIONS as f64;
    println!("  SMSTART+PTRUE+LD1B       {:>9.1} ns/op", ns);
    BenchResult(ns, dur.as_nanos() as i128)
}

fn bench_umopa() -> BenchResult {
    let mut a = [0u8; 32];
    let mut b = [0u8; 32];
    for i in 0..32 {
        a[i] = (i as u8).wrapping_mul(7);
        b[i] = (i as u8).wrapping_mul(13);
    }
    let pa = &a as *const u8;
    let pb = &b as *const u8;

    let (dur, _) = time_it(|| {
        unsafe {
            core::arch::asm!(
                ".arch armv9-a+sme",
                "smstart",
                "ptrue   p0.b",
                "mov     x2, #0",
                "mov     x3, #32",
                "whilelt p1.b, x2, x3",
                "ld1b    {{z0.b}}, p1/z, [{a}]",
                "ld1b    {{z1.b}}, p1/z, [{b}]",
                "umopa   za0.s, p0/m, p0/m, z0.b, z1.b",
                "smstop",
                a = in(reg) pa,
                b = in(reg) pb,
                out("x2") _,
                out("x3") _,
                out("p0") _,
                out("p1") _,
                out("z0") _,
                out("z1") _,
                options(nostack, preserves_flags),
            );
        }
    });
    let ns = dur.as_nanos() as i128 as f64 / ITERATIONS as f64;
    println!("  SME UMOPA total          {:>9.1} ns/op", ns);
    BenchResult(ns, dur.as_nanos() as i128)
}

fn bench_amx() -> BenchResult {
    let a = fuji_sys::fuji_field { limbs: [1u8; 32] };
    let b = fuji_sys::fuji_field { limbs: [1u8; 32] };
    let mut out = fuji_sys::fuji_field { limbs: [0u8; 32] };

    let (dur, _) = time_it(|| {
        unsafe {
            fuji_sys::fuji_f_mul(&a, &b, 0, &mut out);
        }
        black_box(&out);
    });
    let ns = dur.as_nanos() as i128 as f64 / ITERATIONS as f64;
    println!("  AMX field_mul          {:>9.1} ns/op", ns);
    BenchResult(ns, dur.as_nanos() as i128)
}

fn bench_montgomery() -> BenchResult {
    let a = fuji_sys::fuji_field { limbs: [1u8; 32] };
    let b = fuji_sys::fuji_field { limbs: [1u8; 32] };
    let mut out = fuji_sys::fuji_field { limbs: [0u8; 32] };

    let (dur, _) = time_it(|| {
        unsafe {
            fuji_sys::fuji_f_mul_ref(&a, &b, 0, &mut out);
        }
        black_box(&out);
    });
    let ns = dur.as_nanos() as i128 as f64 / ITERATIONS as f64;
    println!("  Montgomery field_mul   {:>9.1} ns/op", ns);
    BenchResult(ns, dur.as_nanos() as i128)
}

// ── Helpers ──────────────────────────────────────────────────

fn time_it(mut f: impl FnMut()) -> (Duration, i128) {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let dur = start.elapsed();
    (dur, dur.as_nanos() as i128)
}
