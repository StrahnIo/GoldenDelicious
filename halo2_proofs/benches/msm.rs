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
}

fn main() {
    for k in [8, 12, 16, 20] {
        let params = Params::<EpAffine>::new(k);
        let n = 1 << k;

        // SW — random scalars
        timed("sw", k, || {
            let coeffs: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();
            let bases = params.get_g();
            let _ = best_multiexp(&coeffs, &bases);
        });

        // SW — all-1 scalars
        timed("sw-all1", k, || {
            let coeffs = vec![Fq::ONE; n];
            let bases = params.get_g();
            let _ = best_multiexp(&coeffs, &bases);
        });

        #[cfg(feature = "fuji")]
        {
            use halo2_proofs::poly::commitment::MSM;

            timed("fuji-prl", k, || {
                let coeffs: Vec<Fq> = vec![Fq::ONE; n];
                let bases = params.get_g();
                let mut msm = MSM::new(&params);
                for (s, base) in coeffs.iter().zip(bases.iter()) {
                    msm.append_term(*s, *base);
                }
                let _ = msm.eval();
            });

            timed("fuji-all1", k, || {
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
            let scalars: Vec<fuji::FujiField> = (0..n).map(|_| fuji::FujiField::one()).collect();
            let _ = fuji::msm::prl_pippenger(&scalars, &bases_mont, curve).unwrap();
            });
        } // cfg block
    }
}
