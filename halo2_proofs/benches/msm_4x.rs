use std::time::Instant;

use ff::{Field, PrimeField};
use halo2_proofs::arithmetic::{best_multiexp, CurveAffine};
use halo2_proofs::pasta::{EpAffine, Fq};
use halo2_proofs::poly::commitment::Params;
use rand_core::OsRng;

fn timed(label: &str, k: u32, f: impl Fn()) {
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    println!("{}/k={:<2}: {:>8.3} ms", label, k, elapsed.as_secs_f64() * 1000.0);
    // Flush to ensure output isn't dropped
    use std::io::Write;
    std::io::stdout().flush().ok();
}

fn main() {
    for k in [11,12,13,14] {
        let start_ = Instant::now();
        println!("Begin Params::<EpAffine>::new(k) param generation for round {}...", k);
        let params = Params::<EpAffine>::new(k);
        let fin_ = start_.elapsed();
        println!("Finished param generation for k = {} in {} ms", k, fin_.as_secs_f64() * 1000.0);
        println!("Casting 1 << k for k = {}...", k);
        let n = 1 << k;
        println!("Starting scalarset and baseset generation...");

        // Generate 4 sets of random scalars + 4 copies of SRS bases
        let coeffs: Vec<Vec<Fq>> = (0..4).map(|_| {
            (0..n).map(|_| Fq::random(OsRng)).collect()
        }).collect();
        println!("Starting base generation...");
        let bases = params.get_g();
        println!("Starting round k = {}!", k);

        // 4× SW — sequential best_multiexp calls (distinct SRS bases)
        timed("sw-4x", k, || {
            let _r0 = best_multiexp(&coeffs[0], &bases);
            let _r1 = best_multiexp(&coeffs[1], &bases);
            let _r2 = best_multiexp(&coeffs[2], &bases);
            let _r3 = best_multiexp(&coeffs[3], &bases);
        });

        // 4× SW — identical G base (like Apple bugrepro), random scalars
        let g_ep = EpAffine::from_xy(
            -pasta_curves::Fp::one(),
            pasta_curves::Fp::from(2u64),
        ).unwrap();
        let ident_bases_ep: Vec<EpAffine> = (0..n).map(|_| g_ep).collect();
        timed("sw-identg-4x", k, || {
            let _r0 = best_multiexp(&coeffs[0], &ident_bases_ep);
            let _r1 = best_multiexp(&coeffs[1], &ident_bases_ep);
            let _r2 = best_multiexp(&coeffs[2], &ident_bases_ep);
            let _r3 = best_multiexp(&coeffs[3], &ident_bases_ep);
        });

        #[cfg(feature = "fuji")]
        {
            use fuji::{FujiAffine, FujiField, FujiCurve};
            let curve = FujiCurve::Pallas;

            // Apple bugrepro identical: identical G base × 4 with random scalars
            let g_aff = fuji::FujiAffine::gen_pallas();
            let g_mont = FujiAffine::from_coordinates(
                g_aff.x().to_mont(curve),
                g_aff.y().to_mont(curve),
            );
            let ident_bases: Vec<FujiAffine> = (0..n).map(|_| g_mont).collect();
            let ident_scalars: Vec<Vec<FujiField>> = coeffs.iter().map(|c| {
                c.iter().map(|s| {
                    let repr = s.to_repr();
                    let mut buf = [0u8; 32];
                    buf.copy_from_slice(repr.as_ref());
                    FujiField::from_bytes(&buf)
                }).collect()
            }).collect();

            timed("prl-identg-4x", k, || {
                let _r0 = fuji::msm::prl_pippenger(&ident_scalars[0], &ident_bases, curve).unwrap();
                let _r1 = fuji::msm::prl_pippenger(&ident_scalars[1], &ident_bases, curve).unwrap();
                let _r2 = fuji::msm::prl_pippenger(&ident_scalars[2], &ident_bases, curve).unwrap();
                let _r3 = fuji::msm::prl_pippenger(&ident_scalars[3], &ident_bases, curve).unwrap();
            });

            // // Pre-convert SRS bases to Mont form (one-time)
            // let bases_mont: Vec<FujiAffine> = bases.iter().map(|base| {
            //     let coords = base.coordinates().unwrap();
            //     let mut xb = [0u8; 32]; xb.copy_from_slice(coords.x().to_repr().as_ref());
            //     let mut yb = [0u8; 32]; yb.copy_from_slice(coords.y().to_repr().as_ref());
            //     FujiAffine::from_coordinates(
            //         FujiField::from_bytes(&xb).to_mont(curve),
            //         FujiField::from_bytes(&yb).to_mont(curve),
            //     )
            // }).collect();

            // // Pre-convert scalars to FujiField (normal form)
            // let scalars_fuji: Vec<Vec<FujiField>> = coeffs.iter().map(|c| {
            //     c.iter().map(|s| {
            //         let repr = s.to_repr();
            //         let mut buf = [0u8; 32];
            //         buf.copy_from_slice(repr.as_ref());
            //         FujiField::from_bytes(&buf)
            //     }).collect()
            // }).collect();

            // // 4× PRL — sequential prl_pippenger calls
            // timed("prl-4x", k, || {
            //     let _r0 = fuji::msm::prl_pippenger(&scalars_fuji[0], &bases_mont, curve).unwrap();
            //     let _r1 = fuji::msm::prl_pippenger(&scalars_fuji[1], &bases_mont, curve).unwrap();
            //     let _r2 = fuji::msm::prl_pippenger(&scalars_fuji[2], &bases_mont, curve).unwrap();
            //     let _r3 = fuji::msm::prl_pippenger(&scalars_fuji[3], &bases_mont, curve).unwrap();
            // });

            // // Batch PRL — single prl_msm_batch call with 4× repeated bases
            // let all_scalars: Vec<FujiField> = scalars_fuji.iter().flat_map(|s| s.iter().copied()).collect();
            // let all_bases: Vec<FujiAffine> = bases_mont.iter().copied().chain(
            //     bases_mont.iter().copied()
            // ).chain(
            //     bases_mont.iter().copied()
            // ).chain(
            //     bases_mont.iter().copied()
            // ).collect();
            // timed("prl-batch-4x", k, || {
            //     let _r = fuji::msm::prl_msm_batch(
            //         &[n as i32, n as i32, n as i32, n as i32],
            //         &all_bases,
            //         &all_scalars,
            //         curve,
            //     ).unwrap();
            // });
        }
    }
}
