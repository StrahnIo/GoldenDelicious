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
    for k in [11] {
        let params = Params::<EpAffine>::new(k);
        let n = 1 << k;

        // Generate 4 sets of random scalars
        let coeffs: Vec<Vec<Fq>> = (0..4)
            .map(|_| (0..n).map(|_| Fq::random(OsRng)).collect())
            .collect();
        let bases = params.get_g();

        // ── SW with identical G base (Pallas generator) ──
        let gen_affine = <pasta_curves::Ep as group::Group>::generator().to_affine();
        let gen_bytes = gen_affine.coordinates().unwrap();
        eprintln!("/// Generator G coordinates (Fp, LE bytes):");
        eprintln!("///   x = {:02x?}", gen_bytes.x().to_repr().as_ref());
        eprintln!("///   y = {:02x?}", gen_bytes.y().to_repr().as_ref());
        let ident_bases_ep: Vec<EpAffine> = (0..n).map(|_| gen_affine).collect();

        // Dump scalars for SW-identg-4x (first 8 of each set)
        for set in 0..4 {
            eprintln!("/// SW-identg-4x scalars set {} (Fq, LE bytes):", set);
            for (j, s) in coeffs[set].iter().take(8).enumerate() {
                eprintln!("///   [{}] = {:02x?}", j, s.to_repr().as_ref());
            }
            eprintln!("///   ... ({} total, showing first 8)", coeffs[set].len());
        }

        // Dump generator base (same for all)
        eprintln!("/// SW-identg-4x base (EpAffine, LE bytes, all identical):");
        eprintln!("///   x = {:02x?}", gen_bytes.x().to_repr().as_ref());
        eprintln!("///   y = {:02x?}", gen_bytes.y().to_repr().as_ref());

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

            // Build Mont-form bases from the SAME normal-form generator as SW
            let mut gx = [0u8; 32]; gx.copy_from_slice(gen_bytes.x().to_repr().as_ref());
            let mut gy = [0u8; 32]; gy.copy_from_slice(gen_bytes.y().to_repr().as_ref());
            let g_mont = FujiAffine::from_coordinates(
                FujiField::from_bytes(&gx).to_mont(curve),
                FujiField::from_bytes(&gy).to_mont(curve),
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

            // Verify: convert SW to Mont and compare against PRL's Mont result bytes
            println!("  verifying prl-identg-4x results...");
            for i in 0..4 {
                let prl_pt = fuji::msm::prl_pippenger(&ident_scalars[i], &ident_bases, curve).unwrap();
                let swc = sw_results[i].coordinates().unwrap();
                // Convert SW normal-form result to Mont for comparison (field_to_fuji + to_mont)
                let sw_x = {
                    let repr = swc.x().to_repr();
                    let mut buf = [0u8; 32];
                    buf.copy_from_slice(repr.as_ref());
                    FujiField::from_bytes(&buf).to_mont(curve)
                };
                let sw_y = {
                    let repr = swc.y().to_repr();
                    let mut buf = [0u8; 32];
                    buf.copy_from_slice(repr.as_ref());
                    FujiField::from_bytes(&buf).to_mont(curve)
                };
                let ok = prl_pt.x_limbs() == sw_x.to_bytes().as_ref()
                    && prl_pt.y_limbs() == sw_y.to_bytes().as_ref();

                if ok {
                    println!("  ✅ poly[{}] MATCH", i);
                } else {
                    println!(
                        "  ❌ MISMATCH poly[{}]: SW ({:02x?}.., {:02x?}..) != PRL ({:02x?}.., {:02x?}..)",
                        i,
                        &sw_x.to_bytes()[..4],
                        &sw_y.to_bytes()[..4],
                        &prl_pt.x_limbs()[..4],
                        &prl_pt.y_limbs()[..4],
                    );
                }
            }
        }
    }
}
