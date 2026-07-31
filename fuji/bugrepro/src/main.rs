// Run s·G through ALL Fuji MSM paths, compare against inputs.md expected result

use fuji::{FujiCurve, FujiField, FujiAffine, FujiPoint};
use fuji::msm::{prl_pippenger, msm_eval, msm_batch};
use fuji::srs::SrsContext;

fn hex_to_bytes(s: &[&str]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, h) in s.iter().enumerate() { out[i] = u8::from_str_radix(h, 16).unwrap(); }
    out
}

fn expected_x() -> [u8; 32] {
    hex_to_bytes(&[
        "87","ba","b3","1a","20","18","be","dc","d3","78","42","62","24","b3","40","38",
        "8e","22","9c","2b","63","46","74","1a","70","f7","05","bc","2d","3f","df","07",
    ])
}
fn expected_y() -> [u8; 32] {
    hex_to_bytes(&[
        "34","73","b6","54","24","f7","8c","4b","6d","75","76","32","ae","a7","75","fe",
        "9a","54","b4","dc","d8","30","f7","4e","5b","3f","03","47","2c","4e","55","26",
    ])
}

fn check(path: &str, pt: FujiPoint, curve: FujiCurve) -> bool {
    let aff = pt.from_mont(curve).to_affine(curve).unwrap();
    let rx = aff.x().to_bytes();
    let ry = aff.y().to_bytes();
    let ok = rx == expected_x() && ry == expected_y();
    eprintln!("  {}: {}", path, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
    if !ok {
        eprintln!("    got X: {:02x?}", &rx[..]);
        eprintln!("    exp X: {:02x?}", &expected_x()[..]);
    }
    ok
}

fn main() {
    let curve = FujiCurve::Pallas;

    // Generator G (normal form LE)
    let gx = hex_to_bytes(&[
        "00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
        "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40",
    ]);
    let gy = hex_to_bytes(&[
        "02","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
        "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
    ]);

    let g_norm = FujiAffine::from_coordinates(
        FujiField::from_bytes(&gx),
        FujiField::from_bytes(&gy),
    );
    let g_mont = FujiAffine::from_coordinates(
        FujiField::from_bytes(&gx).to_mont(curve),
        FujiField::from_bytes(&gy).to_mont(curve),
    );

    // Scalar s (normal form LE)
    let s_bytes = hex_to_bytes(&[
        "83","2f","f0","92","7d","8a","da","ef","3a","e3","a5","12","ff","90","e9","76",
        "02","4f","af","b4","34","70","b5","7b","4e","b0","61","b7","5f","a7","62","1c",
    ]);
    let s_norm = FujiField::from_bytes(&s_bytes);


    let mut all_ok = true;

    // Path 1: prl_pippenger (normal-form scalars, Mont-form bases)
    let r = prl_pippenger(&[s_norm], &[g_mont], curve).unwrap();
    if !check("prl_pippenger", r, curve) { all_ok = false; }

    // Path 2: msm_eval (C library, normal-form inputs)
    let r = msm_eval(&[g_norm], &[s_norm], curve).unwrap();
    if !check("msm_eval", r, curve) { all_ok = false; }

    // Path 3: SRS batch (normal-form scalars)
    let ctx = SrsContext::precompute(&[g_mont], curve).unwrap();
    let mut out = vec![FujiPoint::identity(); 1];
    ctx.commit_batch(&[s_norm], 1, 1, &mut out).unwrap();
    if !check("srs_commit_batch", out[0], curve) { all_ok = false; }

    // Path 4: msm_batch (C library batch, normal-form)
    let r = msm_batch(&[1], &[g_norm], &[s_norm], curve).unwrap();
    if !check("msm_batch", r[0], curve) { all_ok = false; }

    // Path 5: scalar=1 → should give G
    let r = prl_pippenger(&[FujiField::one()], &[g_mont], curve).unwrap();
    let aff = r.from_mont(curve).to_affine(curve).unwrap();
    let gx_expected = hex_to_bytes(&["00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
        "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40"]);
    let one_match = aff.x().to_bytes() == gx_expected;
    eprintln!("  1·G = G: {}", if one_match { "✓" } else { "✗" });
    if !one_match {
        eprintln!("    got: {:02x?}", &aff.x().to_bytes()[..8]);
        eprintln!("    exp: {:02x?}", &gx_expected[..8]);
        all_ok = false;
    }

    // Path 6: prl_pippenger with n=2, compare individually
    let r = prl_pippenger(&[s_norm, FujiField::zero()], &[g_mont, g_mont], curve).unwrap();
    if !check("prl_pippenger + zero scalar", r, curve) { all_ok = false; }

    // Path 7: prl_pippenger with identical base repeated (n=8)
    let bases = vec![g_mont; 8];
    let scalars: Vec<FujiField> = (0..8).map(|i| {
        if i == 0 { s_norm } else { FujiField::zero() }
    }).collect();
    let r = prl_pippenger(&scalars, &bases, curve).unwrap();
    if !check("prl_pippenger + zeros padding", r, curve) { all_ok = false; }

    // Path 8: n×[1] at scale → confirms mont_double_c correctness
    for &n in &[64, 128, 256] {
        let one = FujiField::one();
        let scalars = vec![one; n];
        let bases = vec![g_mont; n];
        let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
        let ref_s = {
            let mut b = [0u8; 32]; b[..8].copy_from_slice(&(n as u64).to_le_bytes());
            FujiField::from_bytes(&b)
        };
        let ref_pt = prl_pippenger(&[ref_s], &[g_mont], curve).unwrap();
        let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
            == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();
        eprintln!("  n×[1] all-G == {}·G (n={}): {}", n, n, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
        all_ok = all_ok && ok;
    }
    // Path 9: deterministic-collision tests (window-0 only, all same base G)
    {
        // Test A: all 64 scalars go to the SAME bucket (value=0x42)
        // → 63 serial mont_mixed_add_c calls (all collisions)
        let n = 64;
        let v = 0x42u8;
        let scalars: Vec<FujiField> = (0..n).map(|_| {
            let mut b = [0u8; 32]; b[0] = v; FujiField::from_bytes(&b)
        }).collect();
        let bases = vec![g_mont; n];
        let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
        let sum_bytes = byte_sum(&scalars);
        let ref_pt = prl_pippenger(&[FujiField::from_bytes(&sum_bytes)], &[g_mont], curve).unwrap();
        let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
            == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();
        eprintln!("  n={} same-bucket: {}", n, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
        all_ok = all_ok && ok;

        // Test B: unique window values 1..=64 → no tails, just bucket reduce
        let n = 64;
        let scalars: Vec<FujiField> = (1..=n).map(|i| {
            let mut b = [0u8; 32]; b[0] = i as u8; FujiField::from_bytes(&b)
        }).collect();
        let bases = vec![g_mont; n];
        let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
        let sum_bytes = byte_sum(&scalars);
        let ref_pt = prl_pippenger(&[FujiField::from_bytes(&sum_bytes)], &[g_mont], curve).unwrap();
        let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
            == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();
        eprintln!("  n={} unique-win: {}", n, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
        all_ok = all_ok && ok;

        // Test C: 32 pairs of same-value scalars → 32 tails, each pair collisions
        let n = 64;
        let scalars: Vec<FujiField> = (0..32).flat_map(|i| {
            let v = (i+1) as u8;
            let mut b1 = [0u8; 32]; b1[0] = v;
            let mut b2 = [0u8; 32]; b2[0] = v;
            [FujiField::from_bytes(&b1), FujiField::from_bytes(&b2)]
        }).collect();
        let bases = vec![g_mont; n];
        let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
        let sum_bytes = byte_sum(&scalars);
        let ref_pt = prl_pippenger(&[FujiField::from_bytes(&sum_bytes)], &[g_mont], curve).unwrap();
        let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
            == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();
        eprintln!("  n={} paired-collisions: {}", n, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
        all_ok = all_ok && ok;

    }

    // Path 10: n=2048 random scalars (final validation)
    {
        use rand_core::RngCore;
        let mut rng = rand_core::OsRng;
        for &n in &[64, 256, 2048] {
            let scalars: Vec<FujiField> = (0..n).map(|_| {
                let mut b = [0u8; 32];
                rng.fill_bytes(&mut b);
                b[31] = 0; b[30] = 0;
                FujiField::from_bytes(&b)
            }).collect();
            let bases = vec![g_mont; n];
            let sum_bytes = byte_sum(&scalars);
            let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
            let ref_pt = prl_pippenger(&[FujiField::from_bytes(&sum_bytes)], &[g_mont], curve).unwrap();
            let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
                == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();
            eprintln!("  n={} random: {}", n, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
            all_ok = all_ok && ok;
        }
    }

    // ── PRL Benchmark (random scalars, identical G bases) ──
    eprintln!("\n=== PRL Pippenger Benchmark (4 threads, random scalars) ===\n");
    eprintln!("  RAYON_NUM_THREADS={:?}", std::env::var("RAYON_NUM_THREADS").unwrap_or("(default)".into()));
    for &k in &[8usize, 11, 12, 13, 14] {
        let n = 1 << k;

        use rand_core::RngCore;
        let mut rng = rand_core::OsRng;
        let scalars: Vec<FujiField> = (0..n).map(|_| {
            let mut b = [0u8; 32];
            rng.fill_bytes(&mut b);
            b[31] = 0; b[30] = 0;
            FujiField::from_bytes(&b)
        }).collect();

        let bases = vec![g_mont; n];

        let sum_bytes = byte_sum(&scalars);
        let ref_pt = prl_pippenger(&[FujiField::from_bytes(&sum_bytes)], &[g_mont], curve).unwrap();

        let warmup = prl_pippenger(&scalars, &bases, curve).unwrap();
        let _ = warmup;

        let start = std::time::Instant::now();
        let trials = if n >= 65536 { 3 } else { 10 };
        for _ in 0..trials {
            let _ = prl_pippenger(&scalars, &bases, curve).unwrap();
        }
        let elapsed = start.elapsed().as_secs_f64() / trials as f64;
        let throughput = (n as f64) / elapsed;

        let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
        let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
            == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();

        eprintln!("  k={:>2}  n={:>6}  {:>8.3} ms  {:>10.0} pts/s  {}",
            k, n, elapsed * 1000.0, throughput,
            if ok { "✓" } else { "✗ MISMATCH" });
        all_ok = all_ok && ok;
    }

    eprintln!();
    if all_ok {
        eprintln!("✓ All paths produce the correct result");
    } else {
        eprintln!("✗ Some paths failed");
        std::process::exit(1);
    }
}

fn byte_sum(scalars: &[FujiField]) -> [u8; 32] {
    let mut s = [0u8; 32];
    for sc in scalars {
        let b = sc.to_bytes();
        let mut carry = 0u16;
        for j in 0..32 {
            let sum = s[j] as u16 + b[j] as u16 + carry;
            s[j] = sum as u8;
            carry = sum >> 8;
        }
    }
    s
}

fn test_n_random(n: usize, curve: FujiCurve, g_mont: &FujiAffine, g_norm: &FujiAffine) -> bool {
    use rand_core::RngCore;
    let mut rng = rand_core::OsRng;

    // Generate n random scalars (use 252-bit values to stay under scalar modulus)
    let scalars: Vec<FujiField> = (0..n)
        .map(|_| {
            let mut b = [0u8; 32];
            rng.fill_bytes(&mut b);
            b[31] &= 0x0f;  // clear top 4 bits → < 2^252 < scalar modulus
            FujiField::from_bytes(&b)
        })
        .collect();

    // All identical Mont-form bases
    let bases = vec![*g_mont; n];

    // Reference: Σ s_i · G via pasta_curves
    use ff::PrimeField;
    use group::{Curve, Group};
    use pasta_curves::arithmetic::CurveAffine;
    use pasta_curves::pallas::{Point, Scalar, Affine};

    let g_ep = Affine::from_xy(
        pasta_curves::Fp::from_repr(g_norm.x().to_bytes()).unwrap(),
        pasta_curves::Fp::from_repr(g_norm.y().to_bytes()).unwrap(),
    ).unwrap();

    let mut expected = Point::identity();
    for s in &scalars {
        let s_fq = Scalar::from_repr(s.to_bytes()).unwrap();
        expected += g_ep * s_fq;
    }

    let expected_xy = expected.to_affine().coordinates().unwrap();
    let expected_x = expected_xy.x().to_repr();

    // PRL Mont-form Pippenger
    let pt = prl_pippenger(&scalars, &bases, curve).unwrap();
    let result_aff = pt.from_mont(curve).to_affine(curve).unwrap();
    let result_x = result_aff.x().to_bytes();

    let ok = result_x[..] == expected_x[..];
    eprintln!("  n={}: {}", n, if ok { "✓ MATCH" } else { "✗ MISMATCH" });
    if !ok {
        eprintln!("    got X: {:02x?}...{:02x?}", &result_x[..4], &result_x[28..]);
        eprintln!("    exp X: {:02x?}...{:02x?}", &expected_x[..4], &expected_x[28..]);
    }
    ok
}
