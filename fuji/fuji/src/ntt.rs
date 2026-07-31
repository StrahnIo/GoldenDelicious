use crate::error::FujiError;
use crate::field::FujiField;
use crate::FujiCurve;

/// In-place NTT over `n` field elements. `n` must be a power of two ≤ 2¹⁶.
///
/// `invert = false` → forward NTT, `invert = true` → inverse NTT.
///
/// # Errors
/// Returns [`FujiError::InvalidContext`] if `n` is not a power of two or > 2¹⁶.
pub fn ntt(a: &mut [FujiField], curve: FujiCurve, invert: bool) -> Result<(), FujiError> {
    let rc = unsafe {
        fuji_sys::fuji_ntt(
            a.as_mut_ptr() as *mut fuji_sys::fuji_field,
            a.len() as i32,
            curve as i32,
            if invert { 1 } else { 0 },
        )
    };
    if rc != 0 { Err(FujiError::from(rc)) } else { Ok(()) }
}

/// Batch NTT: `count` independent transforms of `n` elements each.
///
/// `a` must point to `n * count` elements. Each block of `n` elements is
/// transformed independently.
pub fn ntt_batch(a: &mut [FujiField], n: i32, count: i32, curve: FujiCurve, invert: bool) -> Result<(), FujiError> {
    let rc = unsafe {
        fuji_sys::fuji_ntt_batch(
            a.as_mut_ptr() as *mut fuji_sys::fuji_field,
            n,
            count,
            curve as i32,
            if invert { 1 } else { 0 },
        )
    };
    if rc != 0 { Err(FujiError::from(rc)) } else { Ok(()) }
}
