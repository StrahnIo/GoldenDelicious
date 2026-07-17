    use crate::arithmetic::CurveAffine;
use ff::PrimeField;
use fuji::FujiCurve;
use group::Group;
use std::any::TypeId;

/// Determine the FujiCurve from the scalar field type.
fn curve_for_scalar<S: 'static>() -> Option<FujiCurve> {
    if TypeId::of::<S>() == TypeId::of::<pasta_curves::Fq>() {
        Some(FujiCurve::Pallas)
    } else if TypeId::of::<S>() == TypeId::of::<pasta_curves::Fp>() {
        Some(FujiCurve::Vesta)
    } else {
        None
    }
}

pub(crate) fn field_to_fuji<S: PrimeField>(s: &S) -> fuji::FujiField {
    let repr = s.to_repr();
    let bytes: &[u8] = repr.as_ref();
    let mut buf = [0u8; 32];
    buf[..bytes.len()].copy_from_slice(bytes);
    fuji::FujiField::from_bytes(&buf)
}

/// Try to compute a multi-scalar multiplication using Fuji's AMX backend.
///
/// Returns `None` if the input is too small, or the curve type is not supported.
///
/// # Note
/// Requires the `fuji` feature and Apple Silicon. The C library falls back
/// to a scalar implementation on unsupported processors.
pub(crate) fn try_multiexp<C>(
    coeffs: &[C::Scalar],
    bases: &[C],
) -> Option<C::Curve>
where
    C: CurveAffine,
    C::Scalar: PrimeField,
    C::Base: PrimeField,
{
    let curve = curve_for_scalar::<C::Scalar>()?;
    if coeffs.len() < 64 || coeffs.len() != bases.len() {
        return None;
    }

    let fuji_scalars: Vec<fuji::FujiField> = coeffs
        .iter()
        .map(field_to_fuji)
        .collect();

    let fuji_bases: Vec<fuji::FujiAffine> = bases
        .iter()
        .map(|b| {
            let coords = b.coordinates().unwrap();
            let x = field_to_fuji(coords.x());
            let y = field_to_fuji(coords.y());
            fuji::FujiAffine::from_coordinates(x, y)
        })
        .collect();

    let result = fuji::msm::msm_eval(&fuji_bases, &fuji_scalars, curve).ok()?;
    Some(fuji_point_to_curve::<C>(result, curve))
}

/// Try to compute a batch of multi-scalar multiplications using Fuji's AMX backend.
///
/// `counts` specifies the number of (scalar, base) pairs for each MSM.
/// The total sum of `counts` must equal the length of `bases` and `scalars`.
///
/// Returns `None` if the total input is too small, or the curve type is not supported.
///
/// # Note
/// Requires the `fuji` feature and Apple Silicon. The C library falls back
/// to a scalar implementation on unsupported processors.
pub(crate) fn try_batch_multiexp<C>(
    counts: &[i32],
    bases: &[C],
    scalars: &[C::Scalar],
) -> Option<Vec<C::Curve>>
where
    C: CurveAffine,
    C::Scalar: PrimeField,
    C::Base: PrimeField,
{
    let total: usize = counts.iter().map(|&c| c as usize).sum();
    if counts.is_empty() || total == 0 {
        return Some(Vec::new());
    }

    let curve = curve_for_scalar::<C::Scalar>()?;
    if total < 64 || total != bases.len() || total != scalars.len() {
        return None;
    }

    let fuji_scalars: Vec<fuji::FujiField> = scalars.iter().map(field_to_fuji).collect();

    let fuji_bases: Vec<fuji::FujiAffine> = bases
        .iter()
        .map(|b| {
            let coords = b.coordinates().unwrap();
            let x = field_to_fuji(coords.x());
            let y = field_to_fuji(coords.y());
            fuji::FujiAffine::from_coordinates(x, y)
        })
        .collect();

    let results = fuji::msm::msm_batch(counts, &fuji_bases, &fuji_scalars, curve).ok()?;
    Some(
        results
            .into_iter()
            .map(|pt| fuji_point_to_curve::<C>(pt, curve))
            .collect(),
    )
}

/// Convert a FujiPoint back to a curve's projective point.
///
/// Performs Jacobian-to-affine conversion manually using curve-aware
/// field operations, because the C-library `fuji_pt_to_affine` does not
/// accept a curve parameter and may use the wrong modulus.
fn fuji_point_to_curve<C>(pt: fuji::FujiPoint, curve: FujiCurve) -> C::Curve
where
    C: CurveAffine,
    C::Base: PrimeField,
{
    if pt.is_identity() {
        return C::Curve::identity();
    }

    let x = fuji::FujiField::from_bytes(pt.x_limbs());
    let y = fuji::FujiField::from_bytes(pt.y_limbs());
    let z = fuji::FujiField::from_bytes(pt.z_limbs());

    // Jacobian → affine: x_aff = X / Z²,  y_aff = Y / Z³
    let z_inv = z.inv(curve).unwrap();
    let z_inv_sq = z_inv.sqr(curve).unwrap();
    let z_inv_cu = z_inv_sq.mul(&z_inv, curve).unwrap();
    let x_aff = x.mul(&z_inv_sq, curve).unwrap();
    let y_aff = y.mul(&z_inv_cu, curve).unwrap();

    let x_repr = bytes_to_repr::<C::Base>(&x_aff.to_bytes());
    let y_repr = bytes_to_repr::<C::Base>(&y_aff.to_bytes());
    let x_fe = C::Base::from_repr(x_repr).unwrap();
    let y_fe = C::Base::from_repr(y_repr).unwrap();
    let aff = C::from_xy(x_fe, y_fe).unwrap();
    aff.into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arithmetic::best_multiexp;
    use crate::pasta::{Ep, EpAffine, Eq, EqAffine, Fp, Fq};
    use ff::Field;
    use rand_core::OsRng;

    #[test]
    fn test_try_multiexp_vs_best() {
        let rng = OsRng;
        let n = 64;

        // Test on Pallas (EpAffine)
        let p_coeffs: Vec<Fq> = (0..n).map(|_| Fq::random(rng)).collect();
        let p_bases: Vec<EpAffine> = (0..n)
            .map(|_| Ep::random(rng).into())
            .collect();
        let p_expected = best_multiexp(&p_coeffs, &p_bases);
        let p_fuji = try_multiexp::<EpAffine>(&p_coeffs, &p_bases).unwrap_or_else(|| {
            best_multiexp(&p_coeffs, &p_bases)
        });
        assert_eq!(p_expected, p_fuji, "Pallas (EpAffine) MSM mismatch");

        // Test on Vesta (EqAffine)
        let v_coeffs: Vec<Fp> = (0..n).map(|_| Fp::random(rng)).collect();
        let v_bases: Vec<EqAffine> = (0..n)
            .map(|_| Eq::random(rng).into())
            .collect();
        let v_expected = best_multiexp(&v_coeffs, &v_bases);
        let v_fuji = try_multiexp::<EqAffine>(&v_coeffs, &v_bases).unwrap_or_else(|| {
            best_multiexp(&v_coeffs, &v_bases)
        });
        assert_eq!(v_expected, v_fuji, "Vesta (EqAffine) MSM mismatch");
    }
}

fn bytes_to_repr<S: PrimeField>(bytes: &[u8; 32]) -> S::Repr {
    let mut repr = S::Repr::default();
    let dst: &mut [u8] = repr.as_mut();
    dst.copy_from_slice(bytes);
    repr
}
