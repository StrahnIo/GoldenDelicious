#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::Field;
use halo2_proofs::arithmetic::best_multiexp;
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::Params;
use rand_core::OsRng;

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm");
    for k in 8..13 {
        group
            .bench_function(BenchmarkId::new("k", k), |b| {
                let coeffs = (0..(1 << k)).map(|_| Fq::random(OsRng)).collect::<Vec<_>>();
                let bases = Params::<EpAffine>::new(k).get_g();
                b.iter(|| best_multiexp(&coeffs, &bases))
            })
            .sample_size(30);
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
