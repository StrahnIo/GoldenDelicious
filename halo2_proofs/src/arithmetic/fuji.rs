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
/// Performs Jacobian-to-affine conversion using the base field type directly
/// (avoiding FujiField arithmetic, which may have subtle incompatibilities).
fn fuji_point_to_curve<C>(pt: fuji::FujiPoint, _curve: FujiCurve) -> C::Curve
where
    C: CurveAffine,
    C::Base: PrimeField,
{
    use ff::Field;

    if pt.is_identity() {
        return C::Curve::identity();
    }

    let x_bytes = pt.x_limbs();
    let y_bytes = pt.y_limbs();
    let z_bytes = pt.z_limbs();

    let x = C::Base::from_repr(bytes_to_repr::<C::Base>(x_bytes)).unwrap();
    let y = C::Base::from_repr(bytes_to_repr::<C::Base>(y_bytes)).unwrap();
    let z = C::Base::from_repr(bytes_to_repr::<C::Base>(z_bytes)).unwrap();

    // Jacobian → affine: (X/Z², Y/Z³)
    let z_inv = z.invert().unwrap();
    let z_inv_sq = z_inv.square();
    let z_inv_cu = z_inv_sq * z_inv;
    let x_aff = x * z_inv_sq;
    let y_aff = y * z_inv_cu;

    let aff = C::from_xy(x_aff, y_aff).unwrap();
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
    use crate::arithmetic::best_multiexp;
    use crate::pasta::{Ep, EpAffine, Fq};
    use ff::Field;
    use group::Group;
    use rand_core::OsRng;

    #[test]
    fn debug_fuji_msm_sizes() {
        let rng = OsRng;

        for &n in &[1, 2, 3, 4, 8, 16, 32, 48, 64, 65, 66] {
            let scalars: Vec<Fq> = (0..n).map(|_| Fq::random(rng)).collect();
            let bases: Vec<EpAffine> = (0..n).map(|_| Ep::random(rng).into()).collect();

            let curve = FujiCurve::Pallas;
            let fuji_scalars: Vec<_> = scalars.iter().map(field_to_fuji).collect();
            let fuji_bases: Vec<_> = bases.iter().map(|b| {
                let c = b.coordinates().unwrap();
                fuji::FujiAffine::from_coordinates(field_to_fuji(c.x()), field_to_fuji(c.y()))
            }).collect();

            let sw = best_multiexp(&scalars, &bases);
            let sw_aff = EpAffine::from(sw);
            let sc = sw_aff.coordinates().unwrap();

            match fuji::msm::msm_eval(&fuji_bases, &fuji_scalars, curve) {
                Ok(pt) => {
                    let c = fuji_point_to_curve::<EpAffine>(pt, curve);
                    let c_aff = EpAffine::from(c);
                    let cc = c_aff.coordinates().unwrap();
                    let x_match = cc.x().to_repr() == sc.x().to_repr();
                    let y_match = cc.y().to_repr() == sc.y().to_repr();
                    let ok = x_match && y_match;
                    eprintln!("n={}: {} (x_match={}, y_match={})",
                        n, if ok { "PASS" } else { "FAIL" }, x_match, y_match);
                    if !ok {
                        eprintln!("  fuji x = {:02x?}..", &cc.x().to_repr()[..8]);
                        eprintln!("  sw   x = {:02x?}..", &sc.x().to_repr()[..8]);
                    }
                }
                Err(e) => {
                    eprintln!("n={}: Err({:?})", n, e);
                }
            }
        }
    }
}
