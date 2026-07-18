use crate::arithmetic::CurveAffine;
use ff::PrimeField;
use fuji::{FujiCurve, FujiError};
use group::Group;

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

/// PRL-accelerated Pippenger MSM using port-interleaved mixed addition.
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
    let n = coeffs.len();
    let curve = FujiCurve::Pallas;

    // Convert to Montgomery-form Fuji types.
    let mut scalars = Vec::with_capacity(n);
    for s in coeffs {
        let f = field_to_fuji(s);
        scalars.push(f.to_mont(curve));
    }
    let mut bases_mont = Vec::with_capacity(n);
    for b in bases {
        let coords = b.coordinates().unwrap();
        let x = field_to_fuji(coords.x()).to_mont(curve);
        let y = field_to_fuji(coords.y()).to_mont(curve);
        bases_mont.push(fuji::FujiAffine::from_coordinates(x, y));
    }

    let result = prl_pippenger(&scalars, &bases_mont, curve).ok()?;
    Some(fuji_point_to_curve::<C>(result, curve))
}

// ── Pippenger with PRL bucket fill ─────────────────────────────

fn prl_pippenger(
    scalars: &[fuji::FujiField],
    bases: &[fuji::FujiAffine],
    curve: FujiCurve,
) -> Result<fuji::FujiPoint, FujiError> {
    let n = scalars.len();
    const W: usize = 8;
    const NWIN: usize = 32;
    const NB: usize = 1 << W;

    let mut buckets = vec![fuji::FujiPoint::identity(); NB * NWIN];

    // Pre-extract scalar bytes for window lookups.
    let sbytes: Vec<[u8; 32]> = scalars.iter().map(|s| s.to_bytes()).collect();

    // Bucket fill — process 3 pairs at a time via port interleaving.
    let mut i = 0;
    while i + 3 <= n {
        multi_bucket_add(&sbytes[i..], &bases[i..], &mut buckets, curve)?;
        i += 3;
    }
    while i < n {
        single_bucket_add(&sbytes[i], &bases[i], &mut buckets, curve)?;
        i += 1;
    }

    // Bucket reduction: for each window, sum all buckets.
    let mut winsum = vec![fuji::FujiPoint::identity(); NWIN];
    for win in 0..NWIN {
        let off = win * NB;
        let mut running = fuji::FujiPoint::identity();
        for bv in (1..NB).rev() {
            running = running.add(&buckets[off + bv], curve)?;
            winsum[win] = winsum[win].add(&running, curve)?;
        }
    }

    // Window combination.
    let mut acc = fuji::FujiPoint::identity();
    for win in (0..NWIN).rev() {
        for _ in 0..W {
            acc = acc.double(curve)?;
        }
        acc = acc.add(&winsum[win], curve)?;
    }
    Ok(acc)
}

// ── Per-window bucket addition ──────────────────────────────────

fn multi_bucket_add(
    sbytes: &[[u8; 32]],
    bases: &[fuji::FujiAffine],
    buckets: &mut [fuji::FujiPoint],
    curve: FujiCurve,
) -> Result<(), FujiError> {
    for win in 0..32 {
        let mut idx = [0usize; 3];
        for j in 0..3 {
            idx[j] = window_at(&sbytes[j], win);
        }
        let off = win * 256;
        let b0 = &buckets[off + idx[0]];
        let b1 = &buckets[off + idx[1]];
        let b2 = &buckets[off + idx[2]];
        let (r0, r1, r2) = fuji::FujiPoint::prl_add_mixed_3(
            b0, &bases[0], b1, &bases[1], b2, &bases[2], curve,
        )?;
        buckets[off + idx[0]] = r0;
        buckets[off + idx[1]] = r1;
        buckets[off + idx[2]] = r2;
    }
    Ok(())
}

fn single_bucket_add(
    sbytes: &[u8; 32],
    base: &fuji::FujiAffine,
    buckets: &mut [fuji::FujiPoint],
    curve: FujiCurve,
) -> Result<(), FujiError> {
    for win in 0..32 {
        let idx = window_at(sbytes, win);
        if idx == 0 { continue; }
        let off = win * 256;
        let p = buckets[off + idx].add_mixed(base, curve)?;
        buckets[off + idx] = p;
    }
    Ok(())
}

fn window_at(bytes: &[u8; 32], win: usize) -> usize {
    let skip_bytes = win * 8 / 8;
    let skip_bits = win * 8 - skip_bytes * 8;
    let mut v = 0u64;
    for k in 0..8.min(32 - skip_bytes) {
        v |= (bytes[skip_bytes + k] as u64) << (k * 8);
    }
    ((v >> skip_bits) & 0xFF) as usize
}

// ── Result conversion ───────────────────────────────────────────

fn fuji_point_to_curve<C>(pt: fuji::FujiPoint, curve: FujiCurve) -> C::Curve
where
    C: CurveAffine,
    C::Base: PrimeField,
{
    if pt.is_identity() {
        return C::Curve::identity();
    }
    // The fuji point is in Montgomery form — from_mont then convert.
    let affine = pt.to_affine(curve).unwrap();
    let x = affine.x().from_mont(curve);
    let y = affine.y().from_mont(curve);
    let x_base = C::Base::from_repr(bytes_to_repr::<C::Base>(&x.to_bytes())).unwrap();
    let y_base = C::Base::from_repr(bytes_to_repr::<C::Base>(&y.to_bytes())).unwrap();
    let aff = C::from_xy(x_base, y_base).unwrap();
    aff.into()
}

fn bytes_to_repr<S: PrimeField>(bytes: &[u8; 32]) -> S::Repr {
    let mut repr = S::Repr::default();
    let dst: &mut [u8] = repr.as_mut();
    dst.copy_from_slice(bytes);
    repr
}
