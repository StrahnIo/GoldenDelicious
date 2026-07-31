use crate::error::FujiError;
use crate::curve::{FujiAffine, FujiPoint};
use crate::FujiCurve;

/// Precomputed SRS context for batch polynomial commitments.
pub struct SrsContext {
    ctx: *mut fuji_sys::fuji_srs_ctx,
}

impl std::fmt::Debug for SrsContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrsContext").finish_non_exhaustive()
    }
}

unsafe impl Send for SrsContext {}
unsafe impl Sync for SrsContext {}

impl SrsContext {
    /// Precompute SRS from the generator vector G₀..Gₙ₋₁ in Montgomery form.
    pub fn precompute(bases: &[FujiAffine], curve: FujiCurve) -> Result<Self, FujiError> {
        let n = bases.len() as i32;
        if n == 0 { return Err(FujiError::InvalidContext); }
        let ctx = unsafe {
            fuji_sys::fuji_f_srs_precompute(
                bases.as_ptr() as *const fuji_sys::fuji_affine, n, curve as i32,
            )
        };
        if ctx.is_null() { Err(FujiError::InvalidContext) }
        else { Ok(SrsContext { ctx }) }
    }

    /// Batch-commit: computes ⟨polyⱼ, G⟩ for all j in one call.
    /// `scalars`: flat `n * n_polys` field elements in Mont form.
    /// `out`: `n_polys` result points in Mont form.
    pub fn commit_batch(&self, scalars: &[crate::field::FujiField],
                         n_polys: i32, n: i32,
                         out: &mut [FujiPoint]) -> Result<(), FujiError> {
        if scalars.len() != (n_polys * n) as usize { return Err(FujiError::InvalidContext); }
        if out.len() != n_polys as usize { return Err(FujiError::InvalidContext); }
        let rc = unsafe {
            fuji_sys::fuji_f_srs_commit_batch(
                self.ctx,
                scalars.as_ptr() as *const fuji_sys::fuji_field,
                n_polys, n,
                out.as_mut_ptr() as *mut fuji_sys::fuji_point,
            )
        };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(()) }
    }
}

impl Drop for SrsContext {
    fn drop(&mut self) {
        unsafe { fuji_sys::fuji_f_srs_free(self.ctx); }
    }
}
