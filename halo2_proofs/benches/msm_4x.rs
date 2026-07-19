use std::time::Instant;

use ff::{Field, PrimeField};
use group::{Curve, Group};
use halo2_proofs::arithmetic::{best_multiexp, CurveAffine};
use halo2_proofs::arithmetic::fuji as our_fuji;
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

fn ep_eq(a: &EpAffine, b: &EpAffine) -> bool {
    let ax = a.coordinates().unwrap();
    let bx = b.coordinates().unwrap();
    ax.x().to_repr().as_ref() == bx.x().to_repr().as_ref()
        && ax.y().to_repr().as_ref() == bx.y().to_repr().as_ref()
}

fn main() {
    for k in [11, 12] {
        let params = Params::<EpAffine>::new(k);
        let n = 1 << k;

        // Generate 4 sets of random scalars
        let coeffs: Vec<Vec<Fq>> = (0..4)
            .map(|_| (0..n).map(|_| Fq::random(OsRng)).collect())
            .collect();
        let bases = params.get_g();

        // ── SW with identical G base (Pallas generator) ──
        let gen_affine = <pasta_curves::Ep as group::Group>::generator().to_affine();
        let ident_bases_ep: Vec<EpAffine> = (0..n).map(|_| gen_affine).collect();

        // Compute SW-identg results for verification
        let sw_results: Vec<EpAffine> = (0..4)
            .map(|i| {
                let pt = best_multiexp(&coeffs[i], &ident_bases_ep);
                pt.to_affine()
            })
            .collect();

        // ── Fuji PRL with identical G base ────────────────
        #[cfg(feature = "fuji")]
        {
            use fuji::{FujiAffine, FujiField, FujiCurve};
            use group::Curve;
            let curve = FujiCurve::Pallas;

            let g_aff = fuji::FujiAffine::gen_pallas();
            let g_mont = FujiAffine::from_coordinates(
                g_aff.x().to_mont(curve),
                g_aff.y().to_mont(curve),
            );
            let ident_bases: Vec<FujiAffine> = (0..n).map(|_| g_mont).collect();
            let ident_scalars: Vec<Vec<FujiField>> = coeffs
                .iter()
                .map(|c| {
                    c.iter()
                        .map(|s| {
                            let repr = s.to_repr();
                            let mut buf = [0u8; 32];
                            buf.copy_from_slice(repr.as_ref());
                            FujiField::from_bytes(&buf)
                        })
                        .collect()
                })
                .collect();

            timed("prl-identg-4x", k, || {
                for i in 0..4 {
                    let _ = fuji::msm::prl_pippenger(&ident_scalars[i], &ident_bases, curve).unwrap();
                }
            });

            // Verify each PRL result against SW (compare raw from_mont bytes, like bugrepro)
            println!("  verifying prl-identg-4x results...");
            for i in 0..4 {
                let prl_pt = fuji::msm::prl_pippenger(&ident_scalars[i], &ident_bases, curve).unwrap();
                let prl_norm = prl_pt.from_mont(curve);
                let sw_affine = &sw_results[i];
                let sw_bytes = sw_affine.coordinates().unwrap();

                let ok = prl_norm.x_limbs() == sw_bytes.x().to_repr().as_ref()
                    && prl_norm.y_limbs() == sw_bytes.y().to_repr().as_ref();

                if ok {
                    println!("  ✅ poly[{}] MATCH", i);
                } else {
                    println!(
                        "  ❌ MISMATCH poly[{}]: SW ({:02x?}.., {:02x?}..) != PRL ({:02x?}.., {:02x?}..)",
                        i,
                        &sw_bytes.x().to_repr().as_ref()[..4],
                        &sw_bytes.y().to_repr().as_ref()[..4],
                        &prl_norm.x_limbs()[..4],
                        &prl_norm.y_limbs()[..4],
                    );
                }
            }
        }
    }
}
