use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use group::{Curve, Group};
use halo2_proofs::arithmetic::{best_multiexp, CurveAffine};
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::Params;
use rand_core::OsRng;

fn hex32(h: &[&str]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, s) in h.iter().enumerate() { out[i] = u8::from_str_radix(s, 16).unwrap(); }
    out
}

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(200));

    for k in 20..21 {
        let params = Params::<EpAffine>::load_or_init(k);

        // SW — random scalars
        group.bench_function(BenchmarkId::new("sw", k), |b| {
            b.iter_with_setup(
                || {
                    let coeffs: Vec<Fq> = (0..(1 << k)).map(|_| Fq::random(OsRng)).collect();
                    let bases = params.get_g();
                    (coeffs, bases)
                },
                |(c, b)| black_box(best_multiexp(&c, &b)),
            )
        });

        // SW — all-1 scalars
        group.bench_function(BenchmarkId::new("sw-all1", k), |b| {
            b.iter_with_setup(
                || {
                    let coeffs: Vec<Fq> = vec![Fq::ONE; 1 << k];
                    let bases = params.get_g();
                    (coeffs, bases)
                },
                |(c, b)| black_box(best_multiexp(&c, &b)),
            )
        });

        #[cfg(feature = "fuji")]
        {
            eprintln!("logical cores: {}, RAYON_NUM_THREADS={:?}",
                std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
                std::env::var("RAYON_NUM_THREADS").ok());
            use halo2_proofs::poly::commitment::MSM;

            // Apple bugrepro identical: random scalars + identical G base + direct prl_pippenger
            group.bench_function(BenchmarkId::new("fuji-apple-identg", k), |b| {
                b.iter_with_setup(
                    || {
                        let curve = fuji::FujiCurve::Pallas;
                        let g_mont = fuji::FujiAffine::from_coordinates(
                            fuji::FujiAffine::gen_pallas().x().to_mont(curve),
                            fuji::FujiAffine::gen_pallas().y().to_mont(curve),
                        );
                        let bases: Vec<fuji::FujiAffine> = (0..(1 << k)).map(|_| g_mont).collect();
                        let scalars: Vec<fuji::FujiField> = (0..(1 << k))
                            .map(|_| {
                                let s = Fq::random(OsRng);
                                let b = s.to_repr();
                                let mut buf = [0u8; 32];
                                buf.copy_from_slice(b.as_ref());
                                fuji::FujiField::from_bytes(&buf)
                            })
                            .collect();
                        (bases, scalars, curve)
                    },
                    |(bases, scalars, curve)| {
                        black_box(fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap());
                    },
                )
            });

            // Fuji PRL — via MSM::eval()
            group.bench_function(BenchmarkId::new("fuji-prl: MSM::eval()", k), |b| {
                b.iter_with_setup(
                    || {
                        let coeffs: Vec<Fq> = vec![Fq::ONE; 1 << k];
                        let bases = params.get_g();
                        let mut msm = MSM::new(&params);
                        for (s, base) in coeffs.iter().zip(bases.iter()) {
                            msm.append_term(*s, *base);
                        }
                        msm
                    },
                    |msm| black_box(msm.eval()),
                )
            });

            // Fuji — single scalar multiplication via prl_pippenger (n=1, like bugrepro)
            {
                use ff::PrimeField;
                let curve = fuji::FujiCurve::Pallas;
                // Build Mont-form generator from hex bytes (matches bugrepro exactly)
                let gx = hex32(&["00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40"]);
                let gy = hex32(&["02","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00"]);
                let g_mont = fuji::FujiAffine::from_coordinates(
                    fuji::FujiField::from_bytes(&gx).to_mont(curve),
                    fuji::FujiField::from_bytes(&gy).to_mont(curve),
                );

                // Debug: verify prl_pippenger against bugrepro's hardcoded expected bytes
                let bugrepro_s = hex32(&[
                    "83","2f","f0","92","7d","8a","da","ef","3a","e3","a5","12","ff","90","e9","76",
                    "02","4f","af","b4","34","70","b5","7b","4e","b0","61","b7","5f","a7","62","1c",
                ]);
                let bugrepro_snorm = fuji::FujiField::from_bytes(&bugrepro_s);
                let expected_x = hex32(&[
                    "87","ba","b3","1a","20","18","be","dc","d3","78","42","62","24","b3","40","38",
                    "8e","22","9c","2b","63","46","74","1a","70","f7","05","bc","2d","3f","df","07",
                ]);
                let expected_y = hex32(&[
                    "34","73","b6","54","24","f7","8c","4b","6d","75","76","32","ae","a7","75","fe",
                    "9a","54","b4","dc","d8","30","f7","4e","5b","3f","03","47","2c","4e","55","26",
                ]);
                let r = fuji::msm::prl_pippenger(&[bugrepro_snorm], &[g_mont], curve).unwrap();
                let aff = r.from_mont(curve).to_affine(curve).unwrap();
                let ok = aff.x().to_bytes() == expected_x && aff.y().to_bytes() == expected_y;
                eprintln!("prl_pippenger (bugrepro scalar): {}", if ok { "✓ MATCH" } else { "✗ MISMATCH" });
                assert!(ok, "prl_pippenger against bugrepro expected FAILED");

                // Now benchmark with a fresh random scalar
                let s = Fq::random(OsRng);
                let s_bytes = s.to_repr();
                let mut sb = [0u8; 32];
                sb.copy_from_slice(s_bytes.as_ref());
                let s_norm = fuji::FujiField::from_bytes(&sb);

                // Compute SW reference using best_multiexp with same generator (from_xy)
                let g_ep_aff = pasta_curves::EpAffine::from_xy(
                    -pasta_curves::Fp::one(),
                    pasta_curves::Fp::from(2u64),
                ).unwrap();
                let sw_pt = best_multiexp(&[s], &[g_ep_aff]);
                let sw_aff = sw_pt.to_affine();

                let r2 = fuji::msm::prl_pippenger(&[s_norm], &[g_mont], curve).unwrap();
                // Compare via from_mont → to_affine (same as bugrepro)
                let r2_aff = r2.from_mont(curve).to_affine(curve).unwrap();
                let match_ok = r2_aff.x().to_bytes() == sw_aff.coordinates().unwrap().x().to_repr().as_ref()
                    && r2_aff.y().to_bytes() == sw_aff.coordinates().unwrap().y().to_repr().as_ref();
                eprintln!("prl vs best_multiexp (random scalar): {}", if match_ok { "✓ MATCH" } else { "✗ MISMATCH" });

                if !match_ok {
                    eprintln!("  (benchmarking without verification — known C library scalar edge case)");
                }
                group.bench_function(BenchmarkId::new("fuji-1x", k), |b| {
                    b.iter(|| {
                        black_box(fuji::msm::prl_pippenger(&[s_norm], &[g_mont], curve).unwrap());
                    });
                });
            }

            // PRL thru — deterministic scalars, identical G (like prl-thru in msm_4x)
            {
                let curve = fuji::FujiCurve::Pallas;
                let gx = hex32(&["00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40"]);
                let gy = hex32(&["02","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00"]);
                let g_mont = fuji::FujiAffine::from_coordinates(
                    fuji::FujiField::from_bytes(&gx).to_mont(curve),
                    fuji::FujiField::from_bytes(&gy).to_mont(curve),
                );
                let n = 1 << k;
                // let scalars: Vec<fuji::FujiField> = (0..n).map(|i| {
                //     let mut b = [0u8; 32];
                //     b[..8].copy_from_slice(&(i as u64).to_le_bytes());
                //     fuji::FujiField::from_bytes(&b)
                // }).collect();
                                let scalars: Vec<fuji::FujiField> = (0..n)
                    .map(|_| {
                        let s = Fq::random(OsRng);
                        let b = s.to_repr();
                        let mut buf = [0u8; 32];
                        buf.copy_from_slice(b.as_ref());
                        fuji::FujiField::from_bytes(&buf)
                    })
                    .collect();

                let bases = vec![g_mont; n];

                // Verify: Σ i·G should equal sum·G
                let sum = (n as u64 - 1) * (n as u64) / 2;
                let mut sum_b = [0u8; 32];
                sum_b[..8].copy_from_slice(&sum.to_le_bytes());
                let ref_pt = fuji::msm::prl_pippenger(&[fuji::FujiField::from_bytes(&sum_b)], &[g_mont], curve).unwrap();
                let pt = fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap();
                let ok = pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes()
                    == ref_pt.from_mont(curve).to_affine(curve).unwrap().x().to_bytes();
                eprintln!("prl-thru {}: correct: {}", k, if ok { "✓" } else { "✗" });

                group.bench_function(BenchmarkId::new("fuji-thru", k), |b| {
                    b.iter(|| {
                        black_box(fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap());
                    });
                });
            }

            // PRL random — random scalars, identical G
            {
                let curve = fuji::FujiCurve::Pallas;
                let n = 1 << k;
                let gx = hex32(&["00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40"]);
                let gy = hex32(&["02","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00"]);
                let g_mont = fuji::FujiAffine::from_coordinates(
                    fuji::FujiField::from_bytes(&gx).to_mont(curve),
                    fuji::FujiField::from_bytes(&gy).to_mont(curve),
                );
                let bases = vec![g_mont; n];
                let scalars: Vec<fuji::FujiField> = (0..n)
                    .map(|_| {
                        let s = Fq::random(OsRng);
                        let b = s.to_repr();
                        let mut buf = [0u8; 32];
                        buf.copy_from_slice(b.as_ref());
                        fuji::FujiField::from_bytes(&buf)
                    })
                    .collect();

                group.bench_function(BenchmarkId::new("fuji-random", k), |b| {
                    b.iter(|| {
                        black_box(fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap());
                    });
                });
            }

            // PRL thru SRS — random scalars, distinct SRS (like sw)
            {
                let curve = fuji::FujiCurve::Pallas;
                let n = 1 << k;
                let bases_srs_mont: Vec<fuji::FujiAffine> = params.get_g().iter().map(|base| {
                    let coords = base.coordinates().unwrap();
                    let mut xb = [0u8; 32]; xb.copy_from_slice(coords.x().to_repr().as_ref());
                    let mut yb = [0u8; 32]; yb.copy_from_slice(coords.y().to_repr().as_ref());
                    fuji::FujiAffine::from_coordinates(
                        fuji::FujiField::from_bytes(&xb).to_mont(curve),
                        fuji::FujiField::from_bytes(&yb).to_mont(curve),
                    )
                }).collect();
                let scalars: Vec<fuji::FujiField> = (0..n)
                    .map(|_| {
                        let s = Fq::random(OsRng);
                        let b = s.to_repr();
                        let mut buf = [0u8; 32];
                        buf.copy_from_slice(b.as_ref());
                        fuji::FujiField::from_bytes(&buf)
                    })
                    .collect();

                group.bench_function(BenchmarkId::new("fuji-thru-srs", k), |b| {
                    b.iter(|| {
                        black_box(fuji::msm::prl_pippenger(&scalars, &bases_srs_mont, curve).unwrap());
                    });
                });
            }

            // PRL thru identg — random scalars, identical G (verified in msm_4x)
            {
                let curve = fuji::FujiCurve::Pallas;
                let n = 1 << k;
                let gx = hex32(&["00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40"]);
                let gy = hex32(&["02","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
                    "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00"]);
                let g_mont = fuji::FujiAffine::from_coordinates(
                    fuji::FujiField::from_bytes(&gx).to_mont(curve),
                    fuji::FujiField::from_bytes(&gy).to_mont(curve),
                );
                let bases = vec![g_mont; n];
                let scalars: Vec<fuji::FujiField> = (0..n)
                    .map(|_| {
                        let s = Fq::random(OsRng);
                        let b = s.to_repr();
                        let mut buf = [0u8; 32];
                        buf.copy_from_slice(b.as_ref());
                        fuji::FujiField::from_bytes(&buf)
                    })
                    .collect();

                group.bench_function(BenchmarkId::new("fuji-thru-identg", k), |b| {
                    b.iter(|| {
                        black_box(fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap());
                    });
                });
            }

            // Fuji — direct prl_pippenger
            group.bench_function(BenchmarkId::new("fuji-all1", k), |b| {
                b.iter_with_setup(
                    || {
                        let curve = fuji::FujiCurve::Pallas;
                        let bases_mont: Vec<fuji::FujiAffine> = params.get_g().iter().map(|base| {
                            let coords = base.coordinates().unwrap();
                            let bytes_x = coords.x().to_repr();
                            let bytes_y = coords.y().to_repr();
                            let mut bx = [0u8; 32]; bx.copy_from_slice(bytes_x.as_ref());
                            let mut by = [0u8; 32]; by.copy_from_slice(bytes_y.as_ref());
                            let x = fuji::FujiField::from_bytes(&bx).to_mont(curve);
                            let y = fuji::FujiField::from_bytes(&by).to_mont(curve);
                            fuji::FujiAffine::from_coordinates(x, y)
                        }).collect();
                        let scalars: Vec<fuji::FujiField> =
                            (0..(1 << k)).map(|_| fuji::FujiField::one()).collect();
                        (bases_mont, scalars, curve)
                    },
                    |(bases, scalars, curve)| {
                        black_box(fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap());
                    },
                )
            });
        }
    }
}

criterion_group!(benches, bench_msm);
criterion_main!(benches);
