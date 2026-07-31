use crate::error::FujiError;
use crate::FujiCurve;
use crate::curve::{FujiAffine, FujiPoint};
use crate::field::FujiField;
use rayon::prelude::*;

/// Multi-scalar multiplication: `out = Σᵢ scalars[i] · bases[i]`.
pub fn msm_eval(
    bases: &[FujiAffine],
    scalars: &[FujiField],
    curve: FujiCurve,
) -> Result<FujiPoint, FujiError> {
    if bases.len() != scalars.len() {
        return Err(FujiError::Generic);
    }
    if bases.is_empty() {
        return Ok(FujiPoint::identity());
    }

    let mut out = fuji_sys::fuji_point {
        x: fuji_sys::fuji_field { limbs: [0; 32] },
        y: fuji_sys::fuji_field { limbs: [0; 32] },
        z: fuji_sys::fuji_field { limbs: [0; 32] },
    };

    let rc = unsafe {
        fuji_sys::fuji_msm_eval(
            bases.as_ptr() as *const fuji_sys::fuji_affine,
            scalars.as_ptr() as *const fuji_sys::fuji_field,
            bases.len() as i32,
            curve as i32,
            &mut out,
        )
    };

    if rc != 0 {
        Err(FujiError::from(rc))
    } else {
        Ok(FujiPoint(out))
    }
}

/// Batch multi-scalar multiplication: execute multiple independent MSMs
/// in a single FFI call.
pub fn msm_batch(
    counts: &[i32],
    bases: &[FujiAffine],
    scalars: &[FujiField],
    curve: FujiCurve,
) -> Result<Vec<FujiPoint>, FujiError> {
    let total: i32 = counts.iter().sum();
    if total as usize != bases.len() || total as usize != scalars.len() {
        return Err(FujiError::Generic);
    }
    if counts.is_empty() {
        return Ok(Vec::new());
    }

    let mut out: Vec<fuji_sys::fuji_point> = vec![
        fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        counts.len()
    ];

    let rc = unsafe {
        fuji_sys::fuji_msm_batch(
            counts.as_ptr(),
            counts.len() as i32,
            bases.as_ptr() as *const fuji_sys::fuji_affine,
            scalars.as_ptr() as *const fuji_sys::fuji_field,
            curve as i32,
            out.as_mut_ptr(),
        )
    };

    if rc != 0 {
        Err(FujiError::from(rc))
    } else {
        Ok(out.into_iter().map(FujiPoint).collect())
    }
}

// ── Pure-PRL Mont-form Pippenger ─────────────────────────

const NWIN: usize = 32;

/// Mont-form Pippenger MSM using prl::mul_4x throughout.
///
/// **Bases must already be in Montgomery form.** Returns the result in
/// Montgomery form (call `.from_mont(curve)` to convert to normal).
///
/// Bucket fill uses scan-based packing (O(n), zero Vec<Vec> allocations).
pub fn prl_pippenger(scalars: &[FujiField], bases: &[FujiAffine],
                     curve: FujiCurve) -> Result<FujiPoint, FujiError> {
    let n = scalars.len();
    if std::env::var("FUJI_DEBUG").is_ok() {
        eprintln!("[FUJI_DEBUG] Rust prl_pippenger: n={} bases.len={}", n, bases.len());
        for i in 0..4.min(n) {
            let b = scalars[i].to_bytes();
            eprintln!("  scalar[{}] = {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...",
                i, b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]);
        }
        for i in 0..4.min(bases.len()) {
            let bx = bases[i].x().to_bytes();
            let by = bases[i].y().to_bytes();
            eprintln!("  base[{}].x = {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...  y = {:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}...",
                i, bx[0], bx[1], bx[2], bx[3], bx[4], bx[5], bx[6], bx[7],
                by[0], by[1], by[2], by[3], by[4], by[5], by[6], by[7]);
        }
    }

    let mut buckets = vec![FujiPoint::identity(); 129 * NWIN];
    let sbytes: Vec<[u8; 32]> = scalars.iter().map(|s| s.to_bytes()).collect();

    let winsum_results: Vec<Result<FujiPoint, FujiError>> = buckets.par_chunks_mut(129)
        .enumerate()
        .map(|(win, bkt_slice)| {
            let mut ws = FujiPoint::identity();
            let rc = unsafe {
                fuji_sys::fuji_f_signed_fill_reduce_window(
                    sbytes.as_ptr() as *const u8,
                    bases.as_ptr() as *const fuji_sys::fuji_affine,
                    n as i32, win as i32, curve as i32,
                    bkt_slice.as_mut_ptr() as *mut fuji_sys::fuji_point,
                    &mut ws as *mut FujiPoint as *mut fuji_sys::fuji_point,
                )
            };
            if rc != 0 { return Err(FujiError::from(rc)); }
            Ok(ws)
        })
        .collect();

    let mut winsum = Vec::with_capacity(NWIN);
    for r in winsum_results { winsum.push(r?); }

    let mut acc = FujiPoint::identity();
    let rc = unsafe {
        fuji_sys::fuji_f_combine_windows_32(
            &winsum[0] as *const FujiPoint as *const fuji_sys::fuji_point,
            curve as i32,
            &mut acc as *mut FujiPoint as *mut fuji_sys::fuji_point,
        )
    };
    if rc != 0 { return Err(FujiError::from(rc)); }

    Ok(acc)
}

/// Batch MSM using `prl_pippenger` for each sub-MSM.
pub fn prl_msm_batch(counts: &[i32], all_bases: &[FujiAffine],
                     all_scalars: &[FujiField], curve: FujiCurve
) -> Result<Vec<FujiPoint>, FujiError> {
    let mut results = Vec::with_capacity(counts.len());
    let mut pos = 0usize;
    for &cnt in counts {
        let cnt = cnt as usize;
        let bases = &all_bases[pos..pos + cnt];
        let scalars = &all_scalars[pos..pos + cnt];
        results.push(prl_pippenger(scalars, bases, curve)?);
        pos += cnt;
    }
    Ok(results)
}

/// Batch-4 PRL Pippenger MSM: 4 independent MSMs in a single FFI call.
///
/// `scalars` must be 4 × n elements flat: MSM 0 [0..n-1], MSM 1 [n..2n-1], etc.
/// All 4 MSMs share the same `bases`. Returns 4 results in Montgomery form.
pub fn prl_pippenger_batch_4(
    scalars: &[FujiField],
    bases: &[FujiAffine],
    curve: FujiCurve,
) -> Result<[FujiPoint; 4], FujiError> {
    let n = bases.len();
    if scalars.len() != 4 * n {
        return Err(FujiError::Generic);
    }
    let mut flat = Vec::with_capacity(4 * n * 32);
    for s in scalars {
        flat.extend_from_slice(&s.to_bytes());
    }
    let mut out = [FujiPoint::identity(); 4];
    let rc = unsafe {
        fuji_sys::fuji_f_prl_pippenger_batch_4(
            flat.as_ptr(),
            bases.as_ptr() as *const fuji_sys::fuji_affine,
            n as i32, curve as i32,
            out.as_mut_ptr() as *mut fuji_sys::fuji_point,
        )
    };
    if rc != 0 { return Err(FujiError::from(rc)); }
    Ok(out)
}

/// Identical-G fast-path batch-4 MSM using table lookup.
///
/// All 4 MSMs share the same `base` (normal-form affine). The C library
/// precomputes k×G internally and fills buckets via table lookup —
/// zero field multiplications in the fill phase.
///
/// `scalars` must be 4 × n elements flat: MSM 0 [0..n-1], MSM 1 [n..2n-1], etc.
/// `base` is normal-form affine (not Montgomery). Returns 4 Montgomery-form results.
pub fn prl_fixed_batch_4(
    scalars: &[FujiField],
    base: &FujiAffine,
    n: i32,
    curve: FujiCurve,
) -> Result<[FujiPoint; 4], FujiError> {
    if scalars.len() != 4 * n as usize {
        return Err(FujiError::Generic);
    }
    let mut flat = Vec::with_capacity(4 * n as usize * 32);
    for s in scalars {
        flat.extend_from_slice(&s.to_bytes());
    }
    let base_ptr: *const fuji_sys::fuji_affine = base as *const FujiAffine as *const _;
    let mut out = [FujiPoint::identity(); 4];
    let rc = unsafe {
        fuji_sys::fuji_f_msm_fixed_batch_4(
            flat.as_ptr(),
            base_ptr,
            n, curve as i32,
            out.as_mut_ptr() as *mut fuji_sys::fuji_point,
        )
    };
    if rc != 0 { return Err(FujiError::from(rc)); }
    Ok(out)
}


