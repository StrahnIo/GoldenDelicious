use crate::arithmetic::CurveAffine;
use ff::PrimeField;
use fuji::FujiCurve;
use group::Group;
use std::any::TypeId;

/// Returns `true` if the AMX coprocessor is available at runtime.
pub fn amx_available() -> bool {
    fuji::detection::amx_available()
}

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
/// Returns `None` if AMX is unavailable, the input is too small,
/// or the curve type is not supported.
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
    if !amx_available() || coeffs.len() < 64 || coeffs.len() != bases.len() {
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
/// Returns `None` if AMX is unavailable, the total input is too small,
/// or the curve type is not supported.
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
    if !amx_available() || total < 64 || total != bases.len() || total != scalars.len() {
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
fn fuji_point_to_curve<C>(pt: fuji::FujiPoint, curve: FujiCurve) -> C::Curve
where
    C: CurveAffine,
    C::Base: PrimeField,
{
    if pt.is_identity() {
        return C::Curve::identity();
    }
    let affine = pt.to_affine(curve).unwrap();
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
