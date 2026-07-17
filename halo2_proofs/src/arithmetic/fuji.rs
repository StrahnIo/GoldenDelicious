use crate::arithmetic::CurveAffine;
use fuji::FujiCurve;
use fuji_pasta::{FujiField, PallasTag, VestaTag};
use group::Group;

/// Returns `true` if the AMX coprocessor is available at runtime.
pub fn amx_available() -> bool {
    fuji::detection::amx_available()
}

/// Fuji-accelerated multi-scalar multiplication for Pallas (EpAffine).
pub fn best_multiexp_fuji_pallas(
    coeffs: &[pasta_curves::Fq],
    bases: &[pasta_curves::EpAffine],
) -> Option<pasta_curves::Ep> {
    if !amx_available() || coeffs.len() < 64 || coeffs.len() != bases.len() {
        return None;
    }

    let (fuji_bases, fuji_scalars) = convert_pallas(coeffs, bases);
    let result = fuji::msm::msm_eval(&fuji_bases, &fuji_scalars, FujiCurve::Pallas).ok()?;
    Some(fuji_point_to_ep(result))
}

/// Fuji-accelerated multi-scalar multiplication for Vesta (EqAffine).
pub fn best_multiexp_fuji_vesta(
    coeffs: &[pasta_curves::Fp],
    bases: &[pasta_curves::EqAffine],
) -> Option<pasta_curves::Eq> {
    if !amx_available() || coeffs.len() < 64 || coeffs.len() != bases.len() {
        return None;
    }

    let (fuji_bases, fuji_scalars) = convert_vesta(coeffs, bases);
    let result = fuji::msm::msm_eval(&fuji_bases, &fuji_scalars, FujiCurve::Vesta).ok()?;
    Some(fuji_point_to_eq(result))
}

fn convert_pallas(
    coeffs: &[pasta_curves::Fq],
    bases: &[pasta_curves::EpAffine],
) -> (Vec<fuji::FujiAffine>, Vec<fuji::FujiField>) {
    let fuji_scalars: Vec<fuji::FujiField> = coeffs
        .iter()
        .map(|s| {
            let tagged: FujiField<VestaTag> = (*s).into();
            tagged.into_inner()
        })
        .collect();

    let fuji_bases: Vec<fuji::FujiAffine> = bases
        .iter()
        .map(|b| {
            let coords = b.coordinates().unwrap();
            let x_tagged: FujiField<PallasTag> = (*coords.x()).into();
            let y_tagged: FujiField<PallasTag> = (*coords.y()).into();
            fuji::FujiAffine::from_coordinates(x_tagged.into_inner(), y_tagged.into_inner())
        })
        .collect();

    (fuji_bases, fuji_scalars)
}

fn convert_vesta(
    coeffs: &[pasta_curves::Fp],
    bases: &[pasta_curves::EqAffine],
) -> (Vec<fuji::FujiAffine>, Vec<fuji::FujiField>) {
    let fuji_scalars: Vec<fuji::FujiField> = coeffs
        .iter()
        .map(|s| {
            let tagged: FujiField<PallasTag> = (*s).into();
            tagged.into_inner()
        })
        .collect();

    let fuji_bases: Vec<fuji::FujiAffine> = bases
        .iter()
        .map(|b| {
            let coords = b.coordinates().unwrap();
            let x_tagged: FujiField<VestaTag> = (*coords.x()).into();
            let y_tagged: FujiField<VestaTag> = (*coords.y()).into();
            fuji::FujiAffine::from_coordinates(x_tagged.into_inner(), y_tagged.into_inner())
        })
        .collect();

    (fuji_bases, fuji_scalars)
}

fn fuji_point_to_ep(pt: fuji::FujiPoint) -> pasta_curves::Ep {
    if pt.is_identity() {
        return pasta_curves::Ep::identity();
    }
    let curve = FujiCurve::Pallas;
    let affine = pt.to_affine(curve).unwrap();
    let x_fp: pasta_curves::Fp = {
        let tagged: FujiField<PallasTag> = (*affine.x()).into();
        tagged.into()
    };
    let y_fp: pasta_curves::Fp = {
        let tagged: FujiField<PallasTag> = (*affine.y()).into();
        tagged.into()
    };
    let aff = pasta_curves::EpAffine::from_xy(x_fp, y_fp).unwrap();
    pasta_curves::Ep::from(&aff)
}

fn fuji_point_to_eq(pt: fuji::FujiPoint) -> pasta_curves::Eq {
    if pt.is_identity() {
        return pasta_curves::Eq::identity();
    }
    let curve = FujiCurve::Vesta;
    let affine = pt.to_affine(curve).unwrap();
    let x_fq: pasta_curves::Fq = {
        let tagged: FujiField<VestaTag> = (*affine.x()).into();
        tagged.into()
    };
    let y_fq: pasta_curves::Fq = {
        let tagged: FujiField<VestaTag> = (*affine.y()).into();
        tagged.into()
    };
    let aff = pasta_curves::EqAffine::from_xy(x_fq, y_fq).unwrap();
    pasta_curves::Eq::from(&aff)
}
