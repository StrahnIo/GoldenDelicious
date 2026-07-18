use crate::arithmetic::CurveAffine;
use ff::PrimeField;
use fuji::FujiCurve;
use group::Group;

/// Returns `true` if the Fuji PRL engine is available.
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

/// Try to compute an MSM using Fuji's PRL-accelerated backend.
///
/// Returns `None` if the input is too small or PRL is unavailable.
pub(crate) fn try_multiexp<C>(
    coeffs: &[C::Scalar],
    bases: &[C],
) -> Option<C::Curve>
where
    C: CurveAffine,
    C::Scalar: PrimeField,
    C::Base: PrimeField,
{
    if !fuji_available() || coeffs.len() < 64 || coeffs.len() != bases.len() {
        return None;
    }

    let fuji_scalars: Vec<fuji::FujiField> = coeffs.iter().map(field_to_fuji).collect();
    let fuji_bases: Vec<fuji::FujiAffine> = bases
        .iter()
        .map(|b| {
            let coords = b.coordinates().unwrap();
            let x = field_to_fuji(coords.x());
            let y = field_to_fuji(coords.y());
            fuji::FujiAffine::from_coordinates(x, y)
        })
        .collect();

    let result = fuji::msm::msm_eval(&fuji_bases, &fuji_scalars, FujiCurve::Pallas).ok()?;
    Some(fuji_point_to_curve::<C>(result))
}

/// Convert a FujiPoint back to a curve's projective point.
fn fuji_point_to_curve<C>(pt: fuji::FujiPoint) -> C::Curve
where
    C: CurveAffine,
    C::Base: PrimeField,
{
    if pt.is_identity() {
        return C::Curve::identity();
    }
    let affine = pt.to_affine(FujiCurve::Pallas).unwrap();
    let x = C::Base::from_repr(bytes_to_repr::<C::Base>(&affine.x().to_bytes())).unwrap();
    let y = C::Base::from_repr(bytes_to_repr::<C::Base>(&affine.y().to_bytes())).unwrap();
    let aff = C::from_xy(x, y).unwrap();
    aff.into()
}

fn bytes_to_repr<S: PrimeField>(bytes: &[u8; 32]) -> S::Repr {
    let mut repr = S::Repr::default();
    let dst: &mut [u8] = repr.as_mut();
    dst.copy_from_slice(bytes);
    repr
}
