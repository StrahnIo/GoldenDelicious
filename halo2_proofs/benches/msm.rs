#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use halo2_proofs::arithmetic::{best_multiexp, CurveAffine};
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::Params;
use rand_core::OsRng;

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));

    for k in 11..12 {
        let params = Params::<EpAffine>::new(k);

        // SW — random scalars (realistic)
        group.bench_function(BenchmarkId::new("sw", k), |b| {
            b.iter_with_setup(
                || {
                    let coeffs: Vec<Fq> = (0..(1 << k)).map(|_| Fq::random(OsRng)).collect();
                    let bases = params.get_g();
                    (coeffs, bases)
                },
                |(c, b)| { best_multiexp(&c, &b); },
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
                |(c, b)| { best_multiexp(&c, &b); },
            )
        });

        #[cfg(feature = "fuji")]
        {
            use halo2_proofs::poly::commitment::MSM;

            // Fuji PRL — via MSM::eval() (full integration path)
            group.bench_function(BenchmarkId::new("fuji-prl", k), |b| {
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
                    |msm| { msm.eval(); },
                )
            });

            // Fuji — direct prl_pippenger (pure C, no MSM assembly)
            group.bench_function(BenchmarkId::new("fuji-all1", k), |b| {
                b.iter_with_setup(
                    || {
                        let curve = fuji::FujiCurve::Pallas;
                        let bases_mont: Vec<fuji::FujiAffine> = params.get_g().iter().map(|base| {
                            let coords = base.coordinates().unwrap();
                            let x = {
                                let bytes = coords.x().to_repr();
                                let mut buf = [0u8; 32];
                                buf.copy_from_slice(bytes.as_ref());
                                fuji::FujiField::from_bytes(&buf).to_mont(curve)
                            };
                            let y = {
                                let bytes = coords.y().to_repr();
                                let mut buf = [0u8; 32];
                                buf.copy_from_slice(bytes.as_ref());
                                fuji::FujiField::from_bytes(&buf).to_mont(curve)
                            };
                            fuji::FujiAffine::from_coordinates(x, y)
                        }).collect();
                        let scalars: Vec<fuji::FujiField> = (0..(1 << k)).map(|_| fuji::FujiField::one()).collect();
                        (bases_mont, scalars, curve)
                    },
                    |(bases, scalars, curve)| {
                        let _ = fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap();
                    },
                )
            });
        }
    }
}

criterion_group!(benches, bench_msm);
criterion_main!(benches);
