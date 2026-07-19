use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use halo2_proofs::arithmetic::{best_multiexp, CurveAffine};
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::Params;
use rand_core::OsRng;

fn bench_msm(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(200));

    for k in 11..12 {
        let params = Params::<EpAffine>::new(k);

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
            use halo2_proofs::poly::commitment::MSM;

            // Fuji PRL — via MSM::eval() with random scalars
            group.bench_function(BenchmarkId::new("fuji-prl", k), |b| {
                b.iter_with_setup(
                    || {
                        let coeffs: Vec<Fq> = (0..(1 << k)).map(|_| Fq::random(OsRng)).collect();
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
