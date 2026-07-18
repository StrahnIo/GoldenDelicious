#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::{Field, PrimeField};
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::{Params, MSM};
use rand_core::OsRng;

fn fuji_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm-fuji-prl");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));
    for k in 8..13 {
        group
            .bench_function(BenchmarkId::new("k", k), |b| {
                let params = Params::<EpAffine>::new(k);
                let coeffs: Vec<Fq> = (0..(1 << k)).map(|_| Fq::random(OsRng)).collect();
                let bases = params.get_g();

                b.iter(|| {
                    let mut msm = MSM::new(&params);
                    for (s, base) in coeffs.iter().zip(bases.iter()) {
                        msm.append_term(*s, *base);
                    }
                    msm.eval()
                })
            });
    }
}

criterion_group!(fuji_benches, fuji_benchmark);
criterion_main!(fuji_benches);
