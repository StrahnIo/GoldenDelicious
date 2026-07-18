#[macro_use]
extern crate criterion;

use criterion::{BenchmarkId, Criterion};
use ff::Field;
use halo2_proofs::arithmetic::best_multiexp;
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::{Params, MSM};
use rand_core::OsRng;

fn sw_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm-sw");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));
    for k in 8..13 {
        let coeffs: Vec<Fq> = (0..(1 << k)).map(|_| Fq::random(OsRng)).collect();
        let bases = Params::<EpAffine>::new(k).get_g();
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| best_multiexp(&coeffs, &bases))
        });
    }
}

#[cfg(feature = "fuji")]
fn sw_all1_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm-sw-all1");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));
    for k in 8..13 {
        let coeffs: Vec<Fq> = vec![Fq::ONE; 1 << k];
        let bases = Params::<EpAffine>::new(k).get_g();
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| best_multiexp(&coeffs, &bases))
        });
    }
}

#[cfg(feature = "fuji")]
fn fuji_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("msm-fuji-prl");
    group.sample_size(10).measurement_time(std::time::Duration::from_secs(2)).warm_up_time(std::time::Duration::from_millis(500));
    for k in 8..13 {
        let params = Params::<EpAffine>::new(k);
        // Use all-1 scalars to ensure PRL path is exercised (avoids multi-window bug).
        let coeffs: Vec<Fq> = vec![Fq::ONE; 1 << k];
        let bases = params.get_g();
        let mut msm = MSM::new(&params);
        for (s, base) in coeffs.iter().zip(bases.iter()) {
            msm.append_term(*s, *base);
        }
        group.bench_function(BenchmarkId::new("k", k), |b| {
            b.iter(|| msm.clone().eval())
        });
    }
}

criterion_group!(sw_benches, sw_benchmark);
#[cfg(feature = "fuji")]
criterion_group!(sw_all1_benches, sw_all1_benchmark);
#[cfg(feature = "fuji")]
criterion_group!(fuji_benches, fuji_benchmark);

#[cfg(feature = "fuji")]
criterion_main!(sw_benches, sw_all1_benches, fuji_benches);
#[cfg(not(feature = "fuji"))]
criterion_main!(sw_benches);
