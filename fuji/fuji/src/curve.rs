use crate::error::FujiError;
use crate::{FujiCurve, FujiField};

/// An affine point `(x, y)` on the Pallas or Vesta curve.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FujiAffine(pub(super) fuji_sys::fuji_affine);

/// A Jacobian projective point `(X, Y, Z)` on the Pallas or Vesta curve.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FujiPoint(pub(super) fuji_sys::fuji_point);

impl FujiAffine {
    pub fn gen_pallas() -> Self {
        let mut out = fuji_sys::fuji_affine {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        unsafe { fuji_sys::fuji_gen_pallas(&mut out); }
        FujiAffine(out)
    }

    pub fn gen_vesta() -> Self {
        let mut out = fuji_sys::fuji_affine {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        unsafe { fuji_sys::fuji_gen_vesta(&mut out); }
        FujiAffine(out)
    }

    pub fn from_coordinates(x: FujiField, y: FujiField) -> Self {
        FujiAffine(fuji_sys::fuji_affine { x: x.0, y: y.0 })
    }

    pub fn x(&self) -> &FujiField {
        unsafe { &*(&self.0.x as *const fuji_sys::fuji_field as *const FujiField) }
    }

    pub fn y(&self) -> &FujiField {
        unsafe { &*(&self.0.y as *const fuji_sys::fuji_field as *const FujiField) }
    }
}

impl FujiPoint {
    pub fn x_limbs(&self) -> &[u8; 32] { &self.0.x.limbs }
    pub fn y_limbs(&self) -> &[u8; 32] { &self.0.y.limbs }
    pub fn z_limbs(&self) -> &[u8; 32] { &self.0.z.limbs }

    pub fn identity() -> Self {
        let mut out = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        unsafe { fuji_sys::fuji_pt_id(&mut out); }
        FujiPoint(out)
    }

    pub fn is_identity(&self) -> bool {
        unsafe { fuji_sys::fuji_pt_is_id(&self.0) != 0 }
    }

    pub fn from_affine(a: &FujiAffine, curve: FujiCurve) -> Result<Self, FujiError> {
        let mut out = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        let rc = unsafe { fuji_sys::fuji_pt_from_affine(&a.0, curve as i32, &mut out) };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(FujiPoint(out)) }
    }

    pub fn double(&self, curve: FujiCurve) -> Result<Self, FujiError> {
        let mut out = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        let rc = unsafe { fuji_sys::fuji_pt_double(&self.0, curve as i32, &mut out) };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(FujiPoint(out)) }
    }

    pub fn add(&self, other: &Self, curve: FujiCurve) -> Result<Self, FujiError> {
        let mut out = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        let rc = unsafe { fuji_sys::fuji_pt_add(&self.0, &other.0, curve as i32, &mut out) };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(FujiPoint(out)) }
    }

    pub fn add_mixed(&self, other: &FujiAffine, curve: FujiCurve) -> Result<Self, FujiError> {
        let mut out = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        let rc = unsafe { fuji_sys::fuji_pt_add_mixed(&self.0, &other.0, curve as i32, &mut out) };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(FujiPoint(out)) }
    }

    pub fn to_affine(&self, curve: FujiCurve) -> Result<FujiAffine, FujiError> {
        let mut out = fuji_sys::fuji_affine {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        let rc = unsafe { fuji_sys::fuji_pt_to_affine(&self.0, curve as i32, &mut out) };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(FujiAffine(out)) }
    }

    pub fn to_affine_unchecked(&self, curve: FujiCurve) -> FujiAffine {
        self.to_affine(curve).expect("fuji_pt_to_affine failed")
    }

    pub fn batch_to_affine(points: &[Self], curve: FujiCurve) -> Result<Vec<FujiAffine>, FujiError> {
        if points.is_empty() { return Ok(Vec::new()); }
        let mut out: Vec<fuji_sys::fuji_affine> = vec![
            fuji_sys::fuji_affine { x: fuji_sys::fuji_field { limbs: [0; 32] }, y: fuji_sys::fuji_field { limbs: [0; 32] } };
            points.len()
        ];
        let rc = unsafe {
            fuji_sys::fuji_pt_batch_to_affine(
                points.as_ptr() as *const fuji_sys::fuji_point,
                points.len() as i32, curve as i32, out.as_mut_ptr(),
            )
        };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(out.into_iter().map(FujiAffine).collect()) }
    }

    pub fn negate(&self, curve: FujiCurve) -> Result<Self, FujiError> {
        let mut out = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0; 32] },
            y: fuji_sys::fuji_field { limbs: [0; 32] },
            z: fuji_sys::fuji_field { limbs: [0; 32] },
        };
        let rc = unsafe { fuji_sys::fuji_pt_negate(&self.0, curve as i32, &mut out) };
        if rc != 0 { Err(FujiError::from(rc)) } else { Ok(FujiPoint(out)) }
    }

    pub fn from_projective(x: FujiField, y: FujiField, z: FujiField) -> Self {
        FujiPoint(fuji_sys::fuji_point { x: x.0, y: y.0, z: z.0 })
    }

    /// Convert this point's X, Y, Z from Montgomery form to normal form.
    /// Calls fuji_field_from_mont on each coordinate via FFI.
    pub fn from_mont(&self, curve: FujiCurve) -> Self {
        let to_normal = |f: &fuji_sys::fuji_field| -> fuji_sys::fuji_field {
            let mut out = fuji_sys::fuji_field { limbs: [0u8; 32] };
            unsafe { fuji_sys::fuji_field_from_mont(f, &mut out, curve as i32); }
            out
        };
        Self::from_projective(
            FujiField(to_normal(&self.0.x)),
            FujiField(to_normal(&self.0.y)),
            FujiField(to_normal(&self.0.z)),
        )
    }

    /// Convert this point's X, Y, Z from normal form to Montgomery form.
    pub fn to_mont(&self, curve: FujiCurve) -> Self {
        let to_mont = |f: &fuji_sys::fuji_field| -> fuji_sys::fuji_field {
            let mut out = fuji_sys::fuji_field { limbs: [0u8; 32] };
            unsafe { fuji_sys::fuji_field_to_mont(f, &mut out, curve as i32); }
            out
        };
        Self::from_projective(
            FujiField(to_mont(&self.0.x)),
            FujiField(to_mont(&self.0.y)),
            FujiField(to_mont(&self.0.z)),
        )
    }

    /// Batch convert an array of points from Montgomery form to normal form.
    pub fn from_mont_batch(points: &mut [Self], curve: FujiCurve) {
        for p in points.iter_mut() {
            if !p.is_identity() {
                *p = p.from_mont(curve);
            }
        }
    }

    /// 4-wide batch mixed addition via C library (single FFI call).
    pub fn prl_add_mixed_4(
        a0: &Self, b0: &FujiAffine,
        a1: &Self, b1: &FujiAffine,
        a2: &Self, b2: &FujiAffine,
        a3: &Self, b3: &FujiAffine,
        curve: FujiCurve,
    ) -> Result<(Self, Self, Self, Self), FujiError> {
        if a0.is_identity() || a1.is_identity() || a2.is_identity() || a3.is_identity() {
            return Err(FujiError::InvalidInput);
        }
        let mut out0 = fuji_sys::fuji_point {
            x: fuji_sys::fuji_field { limbs: [0u8; 32] },
            y: fuji_sys::fuji_field { limbs: [0u8; 32] },
            z: fuji_sys::fuji_field { limbs: [0u8; 32] },
        };
        let mut out1 = out0; let mut out2 = out0; let mut out3 = out0;
        let rc = unsafe {
            fuji_sys::fuji_f_point_add_mixed_batch_4(
                &a0.0, &b0.0, &a1.0, &b1.0, &a2.0, &b2.0, &a3.0, &b3.0,
                curve as i32,
                &mut out0, &mut out1, &mut out2, &mut out3,
            )
        };
        if rc != 0 { Err(FujiError::from(rc)) } else {
            Ok((FujiPoint(out0), FujiPoint(out1), FujiPoint(out2), FujiPoint(out3)))
        }
    }
}
