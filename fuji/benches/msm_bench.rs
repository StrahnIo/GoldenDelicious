// ── MSM Benchmarks ─────────────────────────────────────
// Measures prl_pippenger at k=8..15.
// Run: cargo bench --manifest-path rust/Cargo.toml

#[macro_use]
extern crate criterion;

use criterion::{Criterion, black_box, BatchSize};
use fuji::{FujiField, FujiAffine, FujiPoint, FujiCurve, msm};

fn bench_msm(c: &mut Criterion) {
    let ks = [8, 9, 10, 11, 12, 13, 14, 15];

    for &k in &ks {
        let n = 1usize << k;
        let group = format!("k={} (n={})", k, n);

        c.bench_function(&group, |b| {
            // Setup: scalars and bases (reused across iterations)
            let curve = FujiCurve::Pallas;
            let g = FujiAffine::gen_pallas();
            let g_mont = FujiAffine::from_coordinates(
                g.x().to_mont(curve),
                g.y().to_mont(curve),
            );

            let mut bases: Vec<FujiAffine> = Vec::with_capacity(n);
            let mut scalars: Vec<FujiField> = Vec::with_capacity(n);
            for i in 0..n {
                bases.push(g_mont);

                let mut limbs = [0u8; 32];
                for b in 0..32 {
                    limbs[b] = ((i as u64 * 0x9e3779b97f4a7c55u64
                                 + b as u64 * 0xbf58476d1ce4e5b9u64)
                                 >> (b % 8 * 8)) as u8;
                }
                scalars.push(FujiField::from_bytes(&limbs));
            }

            b.iter_batched_ref(
                || (bases.clone(), scalars.clone()),
                |(bases, scalars)| {
                    let result = msm::prl_pippenger(
                        black_box(scalars),
                        black_box(bases),
                        curve,
                    );
                    black_box(result);
                },
                BatchSize::LargeInput,
            );
        });
    }
}

criterion_group!(benches, bench_msm);
criterion_main!(benches);
