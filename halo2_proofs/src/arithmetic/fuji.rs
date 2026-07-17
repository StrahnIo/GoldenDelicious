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
    use ff::{Field, PrimeField};
    use group::{Curve, Group};
    use rand_core::OsRng;

    fn fuji_eval<C: CurveAffine>(bases: &[C], scalars: &[C::Scalar], curve: FujiCurve) -> Option<C::Curve>
    where
        C::Scalar: PrimeField,
        C::Base: PrimeField,
    {
        let fb: Vec<_> = bases.iter().map(|b| {
            let c = b.coordinates().unwrap();
            fuji::FujiAffine::from_coordinates(field_to_fuji(c.x()), field_to_fuji(c.y()))
        }).collect();
        let fs: Vec<_> = scalars.iter().map(field_to_fuji).collect();
        fuji::msm::msm_eval(&fb, &fs, curve).ok().map(|pt| fuji_point_to_curve::<C>(pt, curve))
    }

    fn fuji_batch<C: CurveAffine>(
        counts: &[i32],
        bases: &[C],
        scalars: &[C::Scalar],
        curve: FujiCurve,
    ) -> Option<Vec<C::Curve>>
    where
        C::Scalar: PrimeField,
        C::Base: PrimeField,
    {
        let fb: Vec<_> = bases.iter().map(|b| {
            let c = b.coordinates().unwrap();
            fuji::FujiAffine::from_coordinates(field_to_fuji(c.x()), field_to_fuji(c.y()))
        }).collect();
        let fs: Vec<_> = scalars.iter().map(field_to_fuji).collect();
        fuji::msm::msm_batch(counts, &fb, &fs, curve).ok().map(|pts| {
            pts.into_iter().map(|pt| fuji_point_to_curve::<C>(pt, curve)).collect()
        })
    }

    fn ep_pt_eq(a: &Ep, b: &Ep) -> bool {
        use group::Curve;
        let a_aff = a.to_affine();
        let b_aff = b.to_affine();
        let ca = a_aff.coordinates();
        let cb = b_aff.coordinates();
        if bool::from(ca.is_some()) && bool::from(cb.is_some()) {
            let ca = ca.unwrap();
            let cb = cb.unwrap();
            ca.x().to_repr() == cb.x().to_repr() && ca.y().to_repr() == cb.y().to_repr()
        } else {
            bool::from(ca.is_none()) && bool::from(cb.is_none())
        }
    }

    #[test]
    fn inner_product_batch_vs_single() {
        let mut rng = OsRng;
        let curve = FujiCurve::Pallas;

        for &half in &[1, 2, 4, 8, 16, 32, 64] {
            let total = half * 2;
            let g_prime: Vec<EpAffine> = (0..total).map(|_| Ep::random(&mut rng).into()).collect();
            let p_prime: Vec<Fq> = (0..total).map(|_| Fq::random(&mut rng)).collect();

            // Software ground truth
            let sw_l = best_multiexp(&p_prime[half..], &g_prime[0..half]);
            let sw_r = best_multiexp(&p_prime[0..half], &g_prime[half..]);

            // Two separate fuji_msm_eval calls
            let f_l = fuji_eval::<EpAffine>(&g_prime[0..half], &p_prime[half..], curve);
            let f_r = fuji_eval::<EpAffine>(&g_prime[half..], &p_prime[0..half], curve);

            // One fuji_msm_batch call (inner product ordering)
            let batch_scalars: Vec<Fq> = p_prime[half..].iter().chain(p_prime[0..half].iter()).copied().collect();
            let batch = fuji_batch::<EpAffine>(&[half as i32, half as i32], &g_prime, &batch_scalars, curve);

            let sw_l_ok = true;
            let f_l_ok = f_l.is_some() && ep_pt_eq(&f_l.unwrap(), &sw_l);
            let f_r_ok = f_r.is_some() && ep_pt_eq(&f_r.unwrap(), &sw_r);
            let batch_ok = batch.as_ref().map_or(false, |b| {
                b.len() == 2 && ep_pt_eq(&b[0], &sw_l) && ep_pt_eq(&b[1], &sw_r)
            });

            eprintln!("half={:>2}  sw=ok  two_eval=[{},{}]  batch=[{},{}]",
                half,
                if f_l_ok { "P" } else { "F" },
                if f_r_ok { "P" } else { "F" },
                if batch_ok { "P" } else { "F" },
                if batch_ok { "P" } else { "F" },
            );
        }
    }
}


