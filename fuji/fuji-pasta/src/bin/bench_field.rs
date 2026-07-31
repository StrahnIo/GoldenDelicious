use std::hint::black_box;
use std::time::Instant;

const WARMUP: usize = 2000;
const ITERATIONS: usize = 100000;

fn ns_per_op(label: &str, mut f: impl FnMut()) {
    for _ in 0..WARMUP {
        f();
    }
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        f();
    }
    let ns = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
    println!("  {:<30} {:>8.1} ns/op", label, ns);
}

fn main() {
    println!("=== Field Multiplication Benchmark ===\n");
    println!("CPU: {}", fuji::detection::cpu_brand());
    println!();

    // ── Rust pasta_curves Fp ─────────────────────────────
    {
        use pasta_curves::Fp;

        let a = Fp::from_raw([
            0x73450b1a39dc1315u64,
            0x43e5c11a20e8b4c9u64,
            0x30281b6a191fe4d9u64,
            0x000ac5664325ab8fu64,
        ]);
        let b = Fp::from_raw([
            0x1f68e5601fab11bcu64,
            0x0db509006861e607u64,
            0x7ce96f714193a3c8u64,
            0x5e225bb921b3fef7u64,
        ]);

        println!("Rust pasta_curves::Fp (Montgomery, fully unrolled):");
        ns_per_op("one * one", || {
            black_box(Fp::one() * Fp::one());
        });
        ns_per_op("random mul", || {
            black_box(a * b);
        });
        ns_per_op("self * self (sqr)", || {
            black_box(a * a);
        });
    }

    // ── C fuji_f_mul (AMX dispatch) ─────────────────────
    {
        use fuji_sys::*;

        let a_bytes: [u8; 32] = [
            0x15, 0x13, 0xdc, 0x39, 0x1a, 0x0b, 0x45, 0x73,
            0xc9, 0xb4, 0xe8, 0x20, 0x1a, 0xc1, 0xe5, 0x43,
            0xd9, 0xe4, 0x1f, 0x19, 0x6a, 0x1b, 0x28, 0x30,
            0x8f, 0xab, 0x25, 0x43, 0x66, 0xc5, 0x0a, 0x00,
        ];
        let b_bytes: [u8; 32] = [
            0xbc, 0x11, 0xab, 0x1f, 0x60, 0xe5, 0x68, 0x1f,
            0x07, 0xe6, 0x61, 0x68, 0x00, 0x09, 0xb5, 0x0d,
            0xc8, 0xa3, 0x93, 0x41, 0x71, 0xf4, 0x6e, 0x7c,
            0xf7, 0xfe, 0xb3, 0x21, 0xb9, 0x25, 0x22, 0x5e,
        ];
        let a = fuji_field { limbs: a_bytes };
        let b = fuji_field { limbs: b_bytes };
        let mut out = fuji_field { limbs: [0u8; 32] };

        println!("\nC fuji_f_mul (AMX dispatch, ~2900 ns):");
        ns_per_op("mul", || unsafe {
            black_box(fuji_f_mul(&a, &b, FUJI_CURVE_PALLAS, &mut out));
        });

        println!("\nC fuji_f_mul_ref (scalar Montgomery, ~O2):");
        ns_per_op("mul", || unsafe {
            black_box(fuji_f_mul_ref(&a, &b, FUJI_CURVE_PALLAS, &mut out));
        });

        println!("\nC fuji_f_mul_2x (interleaved pair, ~O3):");
        ns_per_op("mul pair", || unsafe {
            black_box(fuji_f_mul_2x(&a, &b, &a, &b, FUJI_CURVE_PALLAS, &mut out, &mut out));
        });
    }
}
