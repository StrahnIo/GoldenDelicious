/// A 255-bit field element over either the Pallas or Vesta base field.
///
/// Internally stored as 32 little-endian bytes. Arithmetic is handled in C.
/// FFI wrappers for individual field ops (`add`, `sub`, `mul`) are stripped —
/// those operations are rewritten in pure C with inline SME ASM.
use crate::FujiCurve;

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct FujiField(pub fuji_sys::fuji_field);

impl FujiField {
    /// The additive identity (zero).
    pub fn zero() -> Self {
        let mut f = fuji_sys::fuji_field { limbs: [0u8; 32] };
        unsafe { fuji_sys::fuji_f_zero(&mut f); }
        FujiField(f)
    }

    /// The multiplicative identity (one).
    pub fn one() -> Self {
        let mut f = fuji_sys::fuji_field { limbs: [0u8; 32] };
        unsafe { fuji_sys::fuji_f_one(&mut f); }
        FujiField(f)
    }

    /// Construct a field element from 32 little-endian bytes.
    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        let mut f = fuji_sys::fuji_field { limbs: [0u8; 32] };
        unsafe { fuji_sys::fuji_f_from_bytes(bytes.as_ptr(), &mut f); }
        FujiField(f)
    }

    /// Return the canonical little-endian byte representation.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        unsafe { fuji_sys::fuji_f_to_bytes(&self.0, bytes.as_mut_ptr()); }
        bytes
    }

    /// Convert normal form to Montgomery form: `a_mont = a * R mod p`.
    pub fn to_mont(&self, curve: FujiCurve) -> Self {
        let mut out = fuji_sys::fuji_field { limbs: [0u8; 32] };
        unsafe {
            fuji_sys::fuji_field_to_mont(&self.0, &mut out, curve as i32);
        }
        FujiField(out)
    }

    /// Convert Montgomery form to normal form: `a = a_mont * R^(-1) mod p`.
    pub fn from_mont(&self, curve: FujiCurve) -> Self {
        let mut out = fuji_sys::fuji_field { limbs: [0u8; 32] };
        unsafe {
            fuji_sys::fuji_field_from_mont(&self.0, &mut out, curve as i32);
        }
        FujiField(out)
    }
}

impl PartialEq for FujiField {
    fn eq(&self, other: &Self) -> bool {
        self.0.limbs == other.0.limbs
    }
}

impl Eq for FujiField {}
