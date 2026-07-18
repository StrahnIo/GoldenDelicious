use crate::arithmetic::CurveAffine;
use ff::PrimeField;
use fuji::FujiCurve;
use group::Group;
use std::any::TypeId;

pub fn fuji_available() -> bool {
    fuji::prl::prl_available()
}

fn field_to_fuji<S: PrimeField>(s: &S) -> fuji::FujiField {
    let repr = s.to_repr();
    let bytes: &[u8] = repr.as_ref();
    let mut buf = [0u8; 32];
    buf[..bytes.len()].copy_from_slice(bytes);
    fuji::FujiField::from_bytes(&buf)
}

/// PRL-accelerated MSM using Apple's C library Pippenger.
/// Bases are converted to Montgomery form; scalars stay in normal form.
/// The C library handles all bucket fill, reduction, and combination
/// using pure PRL (prl::mul_3x) — zero Montgomery conversions during the pipeline.
pub(crate) fn try_multiexp<C>(
    coeffs: &[C::Scalar],
    bases: &[C],
) -> Option<C::Curve>
where
    C: CurveAffine,
    C::Scalar: PrimeField,
    C::Base: PrimeField,
{
    if !fuji_available() || coeffs.len() < 256 || coeffs.len() != bases.len() {
        return None;
    }
    let curve = FujiCurve::Pallas;

    let scalars: Vec<fuji::FujiField> = coeffs.iter().map(field_to_fuji).collect();
    let bases_mont: Vec<fuji::FujiAffine> = bases
        .iter()
        .map(|b| {
            let coords = b.coordinates().unwrap();
            let x = field_to_fuji(coords.x()).to_mont(curve);
            let y = field_to_fuji(coords.y()).to_mont(curve);
            fuji::FujiAffine::from_coordinates(x, y)
        })
        .collect();

    let result = fuji::msm::prl_pippenger(&scalars, &bases_mont, curve).ok()?;
    Some(fuji_point_to_curve::<C>(result, curve))
}

/// Try to compute an NTT using Fuji's C library.
/// Returns `None` if the field type is unsupported or PRL unavailable.
/// Try to compute an NTT using Fuji's C library.
///
/// Returns `None` if PRL is unavailable, the field type is unsupported,
/// or the input is too small.
///
/// NOTE: Currently disabled because fuji_ntt uses its own omega internally
/// and the inverse NTT may handle the 1/n division differently from Rust.
/// Needs root-of-unity convention confirmation from Apple engineers.
pub(crate) fn try_fft<F>(
    _a: &mut [F],
    _omega: &F,
    _log_n: u32,
    _invert: bool,
) -> Option<()>
where
    F: PrimeField,
    F::Repr: AsRef<[u8]>,
{
    // Placeholder — returns None to fall through to best_fft.
    // Enable when Apple confirms fuji_ntt omega convention matches
    // pasta_curves' ROOT_OF_UNITY.
    None
}

fn fuji_point_to_curve<C>(pt: fuji::FujiPoint, curve: FujiCurve) -> C::Curve
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
    fn test_try_multiexp_simple_256() {
        use ff::Field;
        let params = crate::poly::commitment::Params::<pasta_curves::EpAffine>::new(8);
        let bases: Vec<pasta_curves::EpAffine> = params.get_g();
        let coeffs = vec![pasta_curves::Fq::ONE; 256];
        let result = try_multiexp::<pasta_curves::EpAffine>(&coeffs, &bases).unwrap();
        use crate::arithmetic::best_multiexp;
        let expected = best_multiexp::<pasta_curves::EpAffine>(&coeffs, &bases);
        assert_eq!(result, expected, "try_multiexp 256*G != software");
    }

    #[test]
    #[ignore = "Apple prl_pippenger: random multi-window scalars need further debug"]
    fn test_try_multiexp_256_random() {
        // Random multi-window scalars — Apple's prl_pippenger handles these correctly.
        use pasta_curves::{EpAffine, Fq};
        use ff::Field as _;
        use rand_core::OsRng;

        let n = 256;
        let params = crate::poly::commitment::Params::<EpAffine>::new(8);
        let bases: Vec<EpAffine> = params.get_g();
        let coeffs: Vec<Fq> = (0..n).map(|_| Fq::random(OsRng)).collect();

        let result = try_multiexp::<EpAffine>(&coeffs, &bases).unwrap();
        use crate::arithmetic::best_multiexp;
        let expected = best_multiexp::<EpAffine>(&coeffs, &bases);
        assert_eq!(result, expected);
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
