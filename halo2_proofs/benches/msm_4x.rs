use std::time::Instant;

use ff::{Field, PrimeField};
use group::{Curve, Group};
use halo2_proofs::arithmetic::{best_multiexp, CurveAffine};
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::Params;
use rand_core::OsRng;

fn timed(label: &str, k: u32, f: impl Fn()) {
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    println!("{}/k={:<2}: {:>8.3} ms", label, k, elapsed.as_secs_f64() * 1000.0);
    use std::io::Write;
    std::io::stdout().flush().ok();
}

fn hex32(h: &[&str]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, s) in h.iter().enumerate() { out[i] = u8::from_str_radix(s, 16).unwrap(); }
    out
}

fn main() {
    for k in (10..15) {
        let pb_ = Instant::now();
        let params = Params::<EpAffine>::load_or_init(k);
        let pb_t = pb_.elapsed().as_secs_f64() * 1000.0;
        println!("Params::<EpAffine>::load_or_init({}) took {} millis...", k, pb_t);
        let n = 1 << k;

        // Generate 4 sets of random scalars
        let coeffs: Vec<Vec<Fq>> = (0..4)
            .map(|_| (0..n).map(|_| Fq::random(OsRng)).collect())
            .collect();

        // Identical G base (Pallas generator) for ident-g benchmarks
        let gx = hex32(&["00","00","00","00","ed","30","2d","99","1b","f9","4c","09","fc","98","46","22",
            "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","40"]);
        let gy = hex32(&["02","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00",
            "00","00","00","00","00","00","00","00","00","00","00","00","00","00","00","00"]);
        let g_ep = pasta_curves::EpAffine::from_xy(
            -pasta_curves::Fp::one(),
            pasta_curves::Fp::from(2u64),
        ).unwrap();

        // 4× SW — sequential best_multiexp calls (distinct SRS bases)
        let bases_srs = params.get_g();
        timed("sw-4x", k, || {
            let _r0 = best_multiexp(&coeffs[0], &bases_srs);
            let _r1 = best_multiexp(&coeffs[1], &bases_srs);
            let _r2 = best_multiexp(&coeffs[2], &bases_srs);
            let _r3 = best_multiexp(&coeffs[3], &bases_srs);
        });

        // 4× SW — identical G base, random scalars
        let bases_ident_ep: Vec<EpAffine> = (0..n).map(|_| g_ep).collect();
        timed("sw-identg-4x", k, || {
            let _r0 = best_multiexp(&coeffs[0], &bases_ident_ep);
            let _r1 = best_multiexp(&coeffs[1], &bases_ident_ep);
            let _r2 = best_multiexp(&coeffs[2], &bases_ident_ep);
            let _r3 = best_multiexp(&coeffs[3], &bases_ident_ep);
        });

        #[cfg(feature = "fuji")]
        {
            use fuji::{FujiAffine, FujiField, FujiCurve};
            let curve = FujiCurve::Pallas;

            // Pre-convert scalars to FujiField (normal form)
            let scalars_fuji: Vec<Vec<FujiField>> = coeffs.iter().map(|c| {
                c.iter().map(|s| {
                    let repr = s.to_repr();
                    let mut buf = [0u8; 32];
                    buf.copy_from_slice(repr.as_ref());
                    FujiField::from_bytes(&buf)
                }).collect()
            }).collect();

            // 4× PRL — identical G base (identical bases in Mont form)
            let g_mont = FujiAffine::from_coordinates(
                FujiField::from_bytes(&gx).to_mont(curve),
                FujiField::from_bytes(&gy).to_mont(curve),
            );
            let bases_ident_mont: Vec<FujiAffine> = (0..n).map(|_| g_mont).collect();
            timed("prl-identg-4x", k, || {
                let _r0 = fuji::msm::prl_pippenger(&scalars_fuji[0], &bases_ident_mont, curve).unwrap();
                let _r1 = fuji::msm::prl_pippenger(&scalars_fuji[1], &bases_ident_mont, curve).unwrap();
                let _r2 = fuji::msm::prl_pippenger(&scalars_fuji[2], &bases_ident_mont, curve).unwrap();
                let _r3 = fuji::msm::prl_pippenger(&scalars_fuji[3], &bases_ident_mont, curve).unwrap();
            });

            // 4× PRL — distinct SRS bases
            let bases_srs_mont: Vec<FujiAffine> = bases_srs.iter().map(|base| {
                let coords = base.coordinates().unwrap();
                let mut xb = [0u8; 32]; xb.copy_from_slice(coords.x().to_repr().as_ref());
                let mut yb = [0u8; 32]; yb.copy_from_slice(coords.y().to_repr().as_ref());
                FujiAffine::from_coordinates(
                    FujiField::from_bytes(&xb).to_mont(curve),
                    FujiField::from_bytes(&yb).to_mont(curve),
                )
            }).collect();
            timed("prl-srs-4x", k, || {
                let _r0 = fuji::msm::prl_pippenger(&scalars_fuji[0], &bases_srs_mont, curve).unwrap();
                let _r1 = fuji::msm::prl_pippenger(&scalars_fuji[1], &bases_srs_mont, curve).unwrap();
                let _r2 = fuji::msm::prl_pippenger(&scalars_fuji[2], &bases_srs_mont, curve).unwrap();
                let _r3 = fuji::msm::prl_pippenger(&scalars_fuji[3], &bases_srs_mont, curve).unwrap();
            });

            // Batch PRL — single prl_msm_batch call with repeated bases
            let all_scalars: Vec<FujiField> = scalars_fuji.iter().flat_map(|s| s.iter().copied()).collect();
            let all_bases: Vec<FujiAffine> = bases_srs_mont.iter().copied()
                .chain(bases_srs_mont.iter().copied())
                .chain(bases_srs_mont.iter().copied())
                .chain(bases_srs_mont.iter().copied())
                .collect();
            timed("prl-batch-4x", k, || {
                let _r = fuji::msm::prl_msm_batch(
                    &[n as i32; 4],
                    &all_bases,
                    &all_scalars,
                    curve,
                ).unwrap();
            });
        }
    }
}
