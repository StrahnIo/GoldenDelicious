use crate::arithmetic::CurveAffine;
use ff::PrimeField;
use fuji::{FujiCurve, FujiError};
use group::Group;

pub fn fuji_available() -> bool {
    fuji::prl::prl_available()
}

pub(crate) fn field_to_fuji<S: PrimeField>(s: &S) -> fuji::FujiField {
    let repr = s.to_repr();
    let bytes: &[u8] = repr.as_ref();
    let mut buf = [0u8; 32];
    buf[..bytes.len()].copy_from_slice(bytes);
    fuji::FujiField::from_bytes(&buf)
}

// ── Safe Rust PRL Pippenger ──────────────────────────────
// Uses mont_add/mont_double/prl_add_mixed_4 from the fuji crate.
// Avoids the broken fuji::msm::prl_pippenger C path.

const W: usize = 8;
const NWIN: usize = 32;
const NB: usize = 1 << W;

fn window_at(bytes: &[u8; 32], win: usize) -> usize {
    bytes[win] as usize
}

fn sequential_mixed_add(
    idx: &[usize],
    off: usize,
    bases: &[fuji::FujiAffine],
    buckets: &mut [fuji::FujiPoint],
    curve: FujiCurve,
) -> Result<(), FujiError> {
    for j in 0..idx.len() {
        let bi = idx[j];
        if bi == 0 { continue; }
        let bucket = buckets[off + bi];
        if bucket.is_identity() {
            let z_one = fuji::FujiField::one().to_mont(curve);
            buckets[off + bi] = fuji::FujiPoint::from_projective(
                *bases[j].x(), *bases[j].y(), z_one,
            );
        } else {
            buckets[off + bi] = fuji::curve::mont_mixed_add(&bucket, &bases[j], curve)?;
        }
    }
    Ok(())
}

/// Safe Rust Pippenger using PRL. Avoids the broken C fuji_f_fill_reduce_window.
pub fn prl_pippenger(
    scalars: &[fuji::FujiField],
    bases: &[fuji::FujiAffine],
    curve: FujiCurve,
) -> Result<fuji::FujiPoint, FujiError> {
    let n = scalars.len();
    let mut buckets = vec![fuji::FujiPoint::identity(); NB * NWIN];
    let sbytes: Vec<[u8; 32]> = scalars.iter().map(|s| s.to_bytes()).collect();

    // Bucket fill — 3-wide (pure Rust prl_add_mixed_3), then single.
    let mut i = 0;
    while i + 3 <= n {
        multi_bucket_add_3(&sbytes[i..], &bases[i..], &mut buckets, curve)?;
        i += 3;
    }
    while i < n {
        single_bucket_add(&sbytes[i], &bases[i], &mut buckets, curve)?;
        i += 1;
    }

    // Bucket reduction — pure Rust mont_add (Mont form).
    let mut winsum = vec![fuji::FujiPoint::identity(); NWIN];
    for win in 0..NWIN {
        let off = win * NB;
        let mut running = fuji::FujiPoint::identity();
        for bv in (1..NB).rev() {
            let bkt = buckets[off + bv];
            if !bkt.is_identity() {
                if running.is_identity() {
                    running = bkt;
                } else {
                    running = fuji::curve::mont_add(&running, &bkt, curve)?;
                }
                if winsum[win].is_identity() {
                    winsum[win] = running;
                } else {
                    winsum[win] = fuji::curve::mont_add(&winsum[win], &running, curve)?;
                }
            }
        }
    }

    // Window combination — pure Rust mont_double + mont_add (Mont form).
    let mut acc = fuji::FujiPoint::identity();
    for win in (0..NWIN).rev() {
        for _ in 0..W {
            if !acc.is_identity() {
                acc = fuji::curve::mont_double(&acc, curve)?;
            }
        }
        let ws = winsum[win];
        if !ws.is_identity() {
            if acc.is_identity() {
                acc = ws;
            } else {
                acc = fuji::curve::mont_add(&acc, &ws, curve)?;
            }
        }
    }
    Ok(acc)
}

fn multi_bucket_add_4(
    sbytes: &[[u8; 32]], bases: &[fuji::FujiAffine],
    buckets: &mut [fuji::FujiPoint], curve: FujiCurve,
) -> Result<(), FujiError> {
    for win in 0..32 {
        let mut idx = [0usize; 4];
        for j in 0..4 { idx[j] = window_at(&sbytes[j], win); }
        let off = win * NB;

        let all_safe = idx[0] != 0 && idx[1] != 0 && idx[2] != 0 && idx[3] != 0
            && idx[0] != idx[1] && idx[0] != idx[2] && idx[0] != idx[3]
            && idx[1] != idx[2] && idx[1] != idx[3] && idx[2] != idx[3]
            && !buckets[off + idx[0]].is_identity()
            && !buckets[off + idx[1]].is_identity()
            && !buckets[off + idx[2]].is_identity()
            && !buckets[off + idx[3]].is_identity();

        if all_safe {
            match fuji::FujiPoint::prl_add_mixed_4(
                &buckets[off + idx[0]], &bases[0],
                &buckets[off + idx[1]], &bases[1],
                &buckets[off + idx[2]], &bases[2],
                &buckets[off + idx[3]], &bases[3],
                curve,
            ) {
                Ok((r0, r1, r2, r3)) => {
                    buckets[off + idx[0]] = r0;
                    buckets[off + idx[1]] = r1;
                    buckets[off + idx[2]] = r2;
                    buckets[off + idx[3]] = r3;
                    continue;
                }
                Err(_) => {}
            }
        }
        sequential_mixed_add(&idx, off, bases, buckets, curve)?;
    }
    Ok(())
}

fn multi_bucket_add_3(
    sbytes: &[[u8; 32]], bases: &[fuji::FujiAffine],
    buckets: &mut [fuji::FujiPoint], curve: FujiCurve,
) -> Result<(), FujiError> {
    for win in 0..32 {
        let mut idx = [0usize; 3];
        for j in 0..3 { idx[j] = window_at(&sbytes[j], win); }
        let off = win * NB;

        let all_safe = idx[0] != 0 && idx[1] != 0 && idx[2] != 0
            && idx[0] != idx[1] && idx[0] != idx[2] && idx[1] != idx[2]
            && !buckets[off + idx[0]].is_identity()
            && !buckets[off + idx[1]].is_identity()
            && !buckets[off + idx[2]].is_identity();

        if all_safe {
            match fuji::FujiPoint::prl_add_mixed_3(
                &buckets[off + idx[0]], &bases[0],
                &buckets[off + idx[1]], &bases[1],
                &buckets[off + idx[2]], &bases[2],
                curve,
            ) {
                Ok((r0, r1, r2)) => {
                    buckets[off + idx[0]] = r0;
                    buckets[off + idx[1]] = r1;
                    buckets[off + idx[2]] = r2;
                    continue;
                }
                Err(_) => {}
            }
        }
        sequential_mixed_add(&idx, off, bases, buckets, curve)?;
    }
    Ok(())
}

fn single_bucket_add(
    sbytes: &[u8; 32], base: &fuji::FujiAffine,
    buckets: &mut [fuji::FujiPoint], curve: FujiCurve,
) -> Result<(), FujiError> {
    let z_one = fuji::FujiField::one().to_mont(curve);
    for win in 0..32 {
        let idx = window_at(sbytes, win);
        if idx == 0 { continue; }
        let off = win * NB;
        let bucket = buckets[off + idx];
        if bucket.is_identity() {
            buckets[off + idx] = fuji::FujiPoint::from_projective(
                *base.x(), *base.y(), z_one,
            );
        } else {
            buckets[off + idx] = fuji::curve::mont_mixed_add(&bucket, base, curve)?;
        }
    }
    Ok(())
}

pub(crate) fn fuji_point_to_curve<C>(pt: fuji::FujiPoint, curve: FujiCurve) -> C::Curve
where
    C: CurveAffine,
    C::Base: PrimeField,
{
    if pt.is_identity() {
        return C::Curve::identity();
    }
    // prl_pippenger returns Mont-form — convert to normal before to_affine.
    let pt_norm = pt.from_mont(curve);
    let affine = pt_norm.to_affine(curve).unwrap();
    let x_bytes = affine.x().to_bytes();
    let y_bytes = affine.y().to_bytes();
    let x_base = C::Base::from_repr(bytes_to_repr::<C::Base>(&x_bytes)).unwrap();
    let y_base = C::Base::from_repr(bytes_to_repr::<C::Base>(&y_bytes)).unwrap();
    let aff = C::from_xy(x_base, y_base).unwrap();
    aff.into()
}

fn bytes_to_repr<S: PrimeField>(bytes: &[u8; 32]) -> S::Repr {
    let mut repr = S::Repr::default();
    let dst: &mut [u8] = repr.as_mut();
    dst.copy_from_slice(bytes);
    repr
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eq_g_ep(result: &fuji::FujiPoint, curve: FujiCurve, expected: &pasta_curves::Ep) {
        use ff::PrimeField;
        let pt_norm = result.from_mont(curve);
        let aff = pt_norm.to_affine(curve).unwrap();
        let x = pasta_curves::Fp::from_repr(aff.x().to_bytes()).unwrap();
        let y = pasta_curves::Fp::from_repr(aff.y().to_bytes()).unwrap();
        let aff_ep = pasta_curves::EpAffine::from_xy(x, y).unwrap();
        let got: pasta_curves::Ep = aff_ep.into();
        assert_eq!(got, *expected, "PRL result != software");
    }

    #[test]
    fn test_prl_multiexp_3_pairs() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(
            g_aff.x().to_mont(curve),
            g_aff.y().to_mont(curve),
        );
        let one = fuji::FujiField::one();
        let two = one.add(&one, curve).unwrap();
        let three = two.add(&one, curve).unwrap();

        let bases = [g_mont, g_mont, g_mont];
        let scalars = [one, two, three];
        let result = fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap();

        use crate::arithmetic::best_multiexp;
        use ff::Field as _;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let sfq: [pasta_curves::Fq; 3] = [
            pasta_curves::Fq::ONE,
            pasta_curves::Fq::ONE + pasta_curves::Fq::ONE,
            pasta_curves::Fq::ONE + pasta_curves::Fq::ONE + pasta_curves::Fq::ONE,
        ];
        let bep = [g_ep, g_ep, g_ep];
        let expected = best_multiexp::<pasta_curves::EpAffine>(&sfq, &bep);
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    #[ignore = "Apple prl_pippenger: random multi-window scalars need further debug"]
    fn test_prl_64_random() {
        use pasta_curves::{EpAffine, Fq};
        use ff::Field as _;
        use rand_core::OsRng;

        let curve = FujiCurve::Pallas;
        let n = 64;
        let params = crate::poly::commitment::Params::<EpAffine>::new(6);
        let bases: Vec<EpAffine> = params.get_g();
        let coeffs: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();

        let bases_mont: Vec<fuji::FujiAffine> = bases.iter().map(|b| {
            let c = b.coordinates().unwrap();
            let x = field_to_fuji(c.x()).to_mont(curve);
            let y = field_to_fuji(c.y()).to_mont(curve);
            fuji::FujiAffine::from_coordinates(x, y)
        }).collect();
        let scalars: Vec<fuji::FujiField> = coeffs.iter().map(field_to_fuji).collect();
        let result = fuji::msm::prl_pippenger(&scalars, &bases_mont, curve).unwrap();

        use crate::arithmetic::best_multiexp;
        let expected = best_multiexp::<EpAffine>(&coeffs, &bases);
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    fn test_prl_pippenger_4() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(
            g_aff.x().to_mont(curve),
            g_aff.y().to_mont(curve),
        );
        let one_norm = fuji::FujiField::one();
        let two_norm = one_norm.add(&one_norm, curve).unwrap();
        let three_norm = two_norm.add(&one_norm, curve).unwrap();
        let four_norm = three_norm.add(&one_norm, curve).unwrap();

        let bases = [g_mont, g_mont, g_mont, g_mont];
        let scalars = [one_norm, two_norm, three_norm, four_norm];
        let result = fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap();
        use crate::arithmetic::best_multiexp;
        use ff::Field as _;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let scalars_fq: [pasta_curves::Fq; 4] = [
            pasta_curves::Fq::ONE,
            pasta_curves::Fq::ONE + pasta_curves::Fq::ONE,
            pasta_curves::Fq::ONE + pasta_curves::Fq::ONE + pasta_curves::Fq::ONE,
            pasta_curves::Fq::ONE + pasta_curves::Fq::ONE + pasta_curves::Fq::ONE + pasta_curves::Fq::ONE,
        ];
        let bases_ep = [g_ep, g_ep, g_ep, g_ep];
        let expected = best_multiexp::<pasta_curves::EpAffine>(&scalars_fq, &bases_ep);
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    fn test_prl_pippenger_256_simple_scalars() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(
            g_aff.x().to_mont(curve),
            g_aff.y().to_mont(curve),
        );
        let one_norm = fuji::FujiField::one();
        let scalars = vec![one_norm; 256];
        let bases = vec![g_mont; 256];

        let result = fuji::msm::prl_pippenger(&scalars, &bases, curve).unwrap();
        use crate::arithmetic::best_multiexp;
        use ff::Field as _;
        use group::Curve;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let expected = best_multiexp::<pasta_curves::EpAffine>(
            &vec![pasta_curves::Fq::ONE; 256],
            &vec![g_ep; 256],
        );
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    fn test_prl_pippenger_scalar_256() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(
            g_aff.x().to_mont(curve),
            g_aff.y().to_mont(curve),
        );
        let mut bytes = [0u8; 32];
        bytes[1] = 1;
        let scalar_256 = fuji::FujiField::from_bytes(&bytes);

        let result = fuji::msm::prl_pippenger(&[scalar_256], &[g_mont], curve).unwrap();
        use crate::arithmetic::best_multiexp;
        use ff::Field as _;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let scalar_256_fq = pasta_curves::Fq::from(256u64);
        let expected = best_multiexp::<pasta_curves::EpAffine>(&[scalar_256_fq], &[g_ep]);
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    fn test_prl_pippenger_multi_window_same_bucket() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(
            g_aff.x().to_mont(curve),
            g_aff.y().to_mont(curve),
        );
        let mut bytes = [0u8; 32];
        bytes[1] = 1;
        let s = fuji::FujiField::from_bytes(&bytes);

        let result = fuji::msm::prl_pippenger(&[s, s, s], &[g_mont, g_mont, g_mont], curve).unwrap();
        use crate::arithmetic::best_multiexp;
        use ff::Field as _;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let s_fq = pasta_curves::Fq::from(256u64);
        let expected = best_multiexp::<pasta_curves::EpAffine>(&[s_fq, s_fq, s_fq], &[g_ep, g_ep, g_ep]);
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    fn test_prl_2_window_1() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(g_aff.x().to_mont(curve), g_aff.y().to_mont(curve));
        let mut b1 = [0u8; 32]; b1[1] = 1;
        let mut b2 = [0u8; 32]; b2[1] = 2;
        let s1 = fuji::FujiField::from_bytes(&b1);
        let s2 = fuji::FujiField::from_bytes(&b2);

        let result = fuji::msm::prl_pippenger(&[s1, s2], &[g_mont, g_mont], curve).unwrap();
        use crate::arithmetic::best_multiexp;
        use ff::Field as _;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let expected = best_multiexp::<pasta_curves::EpAffine>(
            &[pasta_curves::Fq::from(256u64), pasta_curves::Fq::from(512u64)],
            &[g_ep, g_ep],
        );
        eq_g_ep(&result, curve, &expected);
    }

    #[test]
    fn test_prl_scalar_257_two_windows() {
        let curve = FujiCurve::Pallas;
        let g_aff = fuji::FujiAffine::gen_pallas();
        let g_mont = fuji::FujiAffine::from_coordinates(g_aff.x().to_mont(curve), g_aff.y().to_mont(curve));
        let mut b = [0u8; 32]; b[0] = 1; b[1] = 1;
        let s = fuji::FujiField::from_bytes(&b);

        let result = fuji::msm::prl_pippenger(&[s], &[g_mont], curve).unwrap();
        use crate::arithmetic::best_multiexp;
        let g_ep = pasta_curves::EpAffine::from_xy(
            pasta_curves::Fp::from_repr(g_aff.x().to_bytes()).unwrap(),
            pasta_curves::Fp::from_repr(g_aff.y().to_bytes()).unwrap(),
        ).unwrap();
        let s257 = { use ff::PrimeField; let mut bb = [0u8; 32]; bb[0] = 1; bb[1] = 1; pasta_curves::Fq::from_repr(bb).unwrap() };
        let expected = best_multiexp::<pasta_curves::EpAffine>(&[s257], &[g_ep]);
        eq_g_ep(&result, curve, &expected);
    }
}
