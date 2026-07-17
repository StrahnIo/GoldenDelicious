#[macro_use]
extern crate criterion;

use crate::arithmetic::best_multiexp;
use crate::pasta::{EqAffine, Fp};
use crate::poly::commitment::Params;
use criterion::{BenchmarkId, Criterion};
use group::ff::Field;
use halo2_proofs::*;
use rand_core::OsRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    for k in 8..13 {
        group
            .bench_function(BenchmarkId::new("k", k), |b| {
                let coeffs = (0..(1 << k)).map(|_| Fp::random(OsRng)).collect::<Vec<_>>();
                let bases = Params::<EqAffine>::new(k).get_g();

                b.iter(|| best_multiexp(&coeffs, &bases))
            })
            .sample_size(30);
    }
}

#[cfg(feature = "fuji")]
fn fuji_benchmark(c: &mut Criterion) {
    use ff::PrimeField;
    use halo2_proofs::arithmetic::CurveAffine;

    let mut group = c.benchmark_group("msm-fuji");
    for k in 8..13 {
        group
            .bench_function(BenchmarkId::new("k", k), |b| {
                let coeffs = (0..(1 << k)).map(|_| Fp::random(OsRng)).collect::<Vec<_>>();
                let bases = Params::<EqAffine>::new(k).get_g();

                let fuji_scalars: Vec<_> = coeffs
                    .iter()
                    .map(|s| {
                        let bytes = s.to_repr();
                        let mut buf = [0u8; 32];
                        buf.copy_from_slice(bytes.as_ref());
                        fuji::FujiField::from_bytes(&buf)
                    })
                    .collect();
                let fuji_bases: Vec<_> = bases
                    .iter()
                    .map(|b| {
                        let coords = b.coordinates().unwrap();
                        let xb = coords.x().to_repr();
                        let yb = coords.y().to_repr();
                        let mut xbuf = [0u8; 32];
                        let mut ybuf = [0u8; 32];
                        xbuf.copy_from_slice(xb.as_ref());
                        ybuf.copy_from_slice(yb.as_ref());
                        fuji::FujiAffine::from_coordinates(
                            fuji::FujiField::from_bytes(&xbuf),
                            fuji::FujiField::from_bytes(&ybuf),
                        )
                    })
                    .collect();

                b.iter(|| {
                    fuji::msm::msm_eval(
                        &fuji_bases,
                        &fuji_scalars,
                        fuji::FujiCurve::Vesta,
                    )
                    .unwrap()
                })
            })
            .sample_size(30);
    }
}

criterion_group!(benches, criterion_benchmark);
#[cfg(feature = "fuji")]
criterion_group!(fuji_benches, fuji_benchmark);

#[cfg(feature = "fuji")]
criterion_main!(benches, fuji_benches);
#[cfg(not(feature = "fuji"))]
criterion_main!(benches);
