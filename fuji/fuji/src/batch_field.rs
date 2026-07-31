use crate::FujiField;
use crate::FujiCurve;

/// Element-wise multiply n pairs: `out[i] = a[i] * b[i]` (Mont-form).
///
/// All inputs/outputs must be in Montgomery form. Uses 4-wide CIOS kernel
/// internally (~17 ns/mul when n ≥ 4). Falls back to scalar for remainder.
///
/// # Panics
///
/// Panics if `a.len() != b.len()` or `a.len() != out.len()`.
pub fn mul(a: &[FujiField], b: &[FujiField], out: &mut [FujiField], curve: FujiCurve) {
    assert_eq!(a.len(), b.len(), "mul_batch: a and b length mismatch");
    assert_eq!(a.len(), out.len(), "mul_batch: output length mismatch");
    let n = a.len() as i32;
    if n == 0 { return; }
    let rc = unsafe {
        fuji_sys::fuji_f_mul_batch(
            a.as_ptr() as *const fuji_sys::fuji_field,
            b.as_ptr() as *const fuji_sys::fuji_field,
            out.as_mut_ptr() as *mut fuji_sys::fuji_field,
            n, curve as i32,
        )
    };
    assert_eq!(rc, 0, "mul_batch failed");
}

/// Element-wise add n pairs: `out[i] = a[i] + b[i]` (Mont-form).
pub fn add(a: &[FujiField], b: &[FujiField], out: &mut [FujiField], curve: FujiCurve) {
    assert_eq!(a.len(), b.len(), "add_batch: a and b length mismatch");
    assert_eq!(a.len(), out.len(), "add_batch: output length mismatch");
    let n = a.len() as i32;
    if n == 0 { return; }
    let rc = unsafe {
        fuji_sys::fuji_f_add_batch(
            a.as_ptr() as *const fuji_sys::fuji_field,
            b.as_ptr() as *const fuji_sys::fuji_field,
            out.as_mut_ptr() as *mut fuji_sys::fuji_field,
            n, curve as i32,
        )
    };
    assert_eq!(rc, 0, "add_batch failed");
}

/// Element-wise sub n pairs: `out[i] = a[i] - b[i]` (Mont-form).
pub fn sub(a: &[FujiField], b: &[FujiField], out: &mut [FujiField], curve: FujiCurve) {
    assert_eq!(a.len(), b.len(), "sub_batch: a and b length mismatch");
    assert_eq!(a.len(), out.len(), "sub_batch: output length mismatch");
    let n = a.len() as i32;
    if n == 0 { return; }
    let rc = unsafe {
        fuji_sys::fuji_f_sub_batch(
            a.as_ptr() as *const fuji_sys::fuji_field,
            b.as_ptr() as *const fuji_sys::fuji_field,
            out.as_mut_ptr() as *mut fuji_sys::fuji_field,
            n, curve as i32,
        )
    };
    assert_eq!(rc, 0, "sub_batch failed");
}

/// Element-wise scale by scalar: `out[i] = a[i] * s` (Mont-form).
pub fn scale(a: &[FujiField], s: &FujiField, out: &mut [FujiField], curve: FujiCurve) {
    assert_eq!(a.len(), out.len(), "scale_batch: output length mismatch");
    let n = a.len() as i32;
    if n == 0 { return; }
    let rc = unsafe {
        fuji_sys::fuji_f_scale_batch(
            a.as_ptr() as *const fuji_sys::fuji_field,
            &s.0 as *const fuji_sys::fuji_field,
            out.as_mut_ptr() as *mut fuji_sys::fuji_field,
            n, curve as i32,
        )
    };
    assert_eq!(rc, 0, "scale_batch failed");
}
