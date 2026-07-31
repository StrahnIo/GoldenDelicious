#![allow(rustdoc::broken_intra_doc_links)]

use std::marker::PhantomData;
use std::ops::{Add, Sub, Mul, Neg, AddAssign, SubAssign, MulAssign};
use std::iter::{Sum, Product};

/* ── CurveTag ────────────────────────────────────────────── */

/// Type-level curve identifier. Implementors are zero-sized marker types.
pub trait CurveTag: Copy + Clone + Send + Sync + 'static {
    fn curve_id() -> i32;
}

/// Marker for the Pallas curve.
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct PallasTag;
/// Marker for the Vesta curve.
#[derive(Copy, Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct VestaTag;

impl CurveTag for PallasTag { fn curve_id() -> i32 { 0 } }
impl CurveTag for VestaTag  { fn curve_id() -> i32 { 1 } }

/// Associates the scalar curve tag for a given curve (Pasta cycle property).
pub trait CurveScalar: CurveTag {
    type Scalar: CurveTag;
}
impl CurveScalar for PallasTag { type Scalar = VestaTag; }
impl CurveScalar for VestaTag  { type Scalar = PallasTag; }

/// Per-curve field constants (for PrimiField impl).
pub trait CurveConstants: CurveScalar {
    const MODULUS_STR: &'static str;
    const TWO_INV_BYTES: [u8; 32];
    const MODULUS_BYTES: [u8; 32];
}
impl CurveConstants for PallasTag {
    const MODULUS_STR: &'static str = "0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001";
    const MODULUS_BYTES: [u8; 32] = [
        0x01, 0x00, 0x00, 0x00, 0xed, 0x30, 0x2d, 0x99,
        0x1b, 0xf9, 0x4c, 0x09, 0xfc, 0x98, 0x46, 0x22,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];
    const TWO_INV_BYTES: [u8; 32] = [
        0x01, 0x00, 0x00, 0x80, 0x76, 0x98, 0x96, 0xcc,
        0x8d, 0x7c, 0xa6, 0x04, 0x7e, 0x4c, 0x23, 0x11,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
    ];
}
impl CurveConstants for VestaTag {
    const MODULUS_STR: &'static str = "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001";
    const MODULUS_BYTES: [u8; 32] = [
        0x01, 0x00, 0x00, 0x00, 0x21, 0xeb, 0x46, 0x8c,
        0xdd, 0xa8, 0x94, 0x09, 0xfc, 0x98, 0x46, 0x22,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40,
    ];
    const TWO_INV_BYTES: [u8; 32] = [
        0x01, 0x00, 0x00, 0x80, 0x90, 0x75, 0x23, 0xc6,
        0x6e, 0x54, 0xca, 0x04, 0x7e, 0x4c, 0x23, 0x11,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x20,
    ];
}

/* ── FujiField<C> ─────────────────────────────────────────── */

/// A Pasta field element tagged with its curve.
///
/// Wraps [`fuji::FujiField`] with a [`CurveTag`] phantom parameter so the
/// curve is tracked at the type level. Arithmetic operators use the tag's
/// [`curve_id`](CurveTag::curve_id) to select the correct modulus.
#[repr(transparent)]
pub struct FujiField<C: CurveTag>(fuji::FujiField, PhantomData<C>);

impl<C: CurveTag> FujiField<C> {
    pub fn zero() -> Self { FujiField(fuji::FujiField::zero(), PhantomData) }
    pub fn one()  -> Self { FujiField(fuji::FujiField::one(),  PhantomData) }

    pub fn from_bytes(bytes: &[u8; 32]) -> Self {
        FujiField(fuji::FujiField::from_bytes(bytes), PhantomData)
    }

    pub fn to_bytes(&self) -> [u8; 32] { self.0.to_bytes() }

    pub fn inner(&self) -> &fuji::FujiField { &self.0 }
    pub fn into_inner(self) -> fuji::FujiField { self.0 }

    fn curve() -> fuji::FujiCurve {
        match C::curve_id() {
            0 => fuji::FujiCurve::Pallas,
            _ => fuji::FujiCurve::Vesta,
        }
    }

    pub fn add(&self, other: Self) -> Self {
        FujiField(self.0.add(&other.0, Self::curve()).unwrap(), PhantomData)
    }

    pub fn sub(&self, other: Self) -> Self {
        FujiField(self.0.sub(&other.0, Self::curve()).unwrap(), PhantomData)
    }

    pub fn mul(&self, other: Self) -> Self {
        FujiField(self.0.mul(&other.0, Self::curve()).unwrap(), PhantomData)
    }

    pub fn square(&self) -> Self {
        FujiField(self.0.sqr(Self::curve()).unwrap(), PhantomData)
    }

    pub fn double(&self) -> Self { self.add(*self) }

    pub fn neg(&self) -> Self {
        let zero = fuji::FujiField::zero();
        FujiField(self.0.sub(&zero, Self::curve()).unwrap(), PhantomData)
    }

    pub fn invert(&self) -> subtle::CtOption<Self> {
        match self.0.inv(Self::curve()) {
            Ok(f) => subtle::CtOption::new(FujiField(f, PhantomData), 1u8.into()),
            Err(_) => subtle::CtOption::new(Self::zero(), 0u8.into()),
        }
    }

    pub fn sqrt(&self) -> subtle::CtOption<Self> {
        match self.0.sqrt(Self::curve()) {
            Ok(f) => subtle::CtOption::new(FujiField(f, PhantomData), 1u8.into()),
            Err(_) => subtle::CtOption::new(Self::zero(), 0u8.into()),
        }
    }

    pub fn is_zero(&self) -> subtle::Choice { subtle::Choice::from(self.0.eq(&fuji::FujiField::zero()) as u8) }
    pub fn eq(&self, other: &Self) -> bool { self.0.eq(&other.0) }
    pub fn is_odd(&self) -> subtle::Choice { subtle::Choice::from(self.0.to_bytes()[0] & 1) }
}

impl<C: CurveTag> Clone for FujiField<C> {
    fn clone(&self) -> Self { *self }
}
impl<C: CurveTag> Copy for FujiField<C> {}
impl<C: CurveTag> PartialEq for FujiField<C> {
    fn eq(&self, other: &Self) -> bool { self.0.eq(&other.0) }
}
impl<C: CurveTag> Eq for FujiField<C> {}

impl<C: CurveTag> std::fmt::Debug for FujiField<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FujiField({:02x?})", &self.0.to_bytes())
    }
}

impl<C: CurveTag> From<FujiField<C>> for fuji::FujiField {
    fn from(f: FujiField<C>) -> Self { f.0 }
}
impl<C: CurveTag> From<fuji::FujiField> for FujiField<C> {
    fn from(f: fuji::FujiField) -> Self { FujiField(f, PhantomData) }
}
impl<C: CurveTag> Default for FujiField<C> {
    fn default() -> Self { Self::zero() }
}

/* ── FujiAffine<C> + FujiPoint<C> ─────────────────────────── */

/// Affine point tagged with its curve. Wraps [`fuji::FujiAffine`].
#[repr(transparent)]
pub struct FujiAffine<C: CurveTag>(fuji::FujiAffine, PhantomData<C>);

/// Jacobian projective point tagged with its curve. Wraps [`fuji::FujiPoint`].
#[repr(transparent)]
pub struct FujiPoint<C: CurveTag>(fuji::FujiPoint, PhantomData<C>);

impl<C: CurveTag> FujiAffine<C> {
    pub fn gen_pallas() -> Self { FujiAffine(fuji::FujiAffine::gen_pallas(), PhantomData) }
    pub fn gen_vesta() -> Self   { FujiAffine(fuji::FujiAffine::gen_vesta(), PhantomData) }
    pub fn inner(&self) -> &fuji::FujiAffine { &self.0 }
    pub fn into_inner(self) -> fuji::FujiAffine { self.0 }
}

impl<C: CurveTag> FujiPoint<C> {
    fn curve() -> fuji::FujiCurve {
        match C::curve_id() { 0 => fuji::FujiCurve::Pallas, _ => fuji::FujiCurve::Vesta }
    }

    pub fn identity() -> Self {
        FujiPoint(fuji::FujiPoint::identity(), PhantomData)
    }

    pub fn from_affine(a: &FujiAffine<C>) -> Self {
        let p = fuji::FujiPoint::from_affine(&a.0, Self::curve()).unwrap();
        FujiPoint(p, PhantomData)
    }

    pub fn double(&self) -> Self {
        FujiPoint(self.0.double(Self::curve()).unwrap(), PhantomData)
    }

    pub fn add(&self, other: &Self) -> Self {
        FujiPoint(self.0.add(&other.0, Self::curve()).unwrap(), PhantomData)
    }

    pub fn add_mixed(&self, other: &FujiAffine<C>) -> Self {
        FujiPoint(self.0.add_mixed(&other.0, Self::curve()).unwrap(), PhantomData)
    }

    pub fn neg(&self) -> Self {
        FujiPoint(self.0.negate(Self::curve()).unwrap(), PhantomData)
    }

    pub fn is_identity(&self) -> bool { self.0.is_identity() }

    pub fn inner(&self) -> &fuji::FujiPoint { &self.0 }
    pub fn into_inner(self) -> fuji::FujiPoint { self.0 }

    /// Convert to affine coordinates.
    pub fn to_affine(&self) -> fuji::FujiAffine {
        self.0.to_affine_unchecked(Self::curve())
    }

    /// Batch convert Jacobian points to affine coordinates.
    pub fn batch_to_affine(points: &[Self]) -> Vec<fuji::FujiAffine> {
        let inner: Vec<fuji::FujiPoint> = points.iter().map(|p| p.0).collect();
        fuji::FujiPoint::batch_to_affine(&inner, Self::curve())
            .expect("batch_to_affine failed")
    }

    /// Batch MSM: execute multiple independent MSMs using PRL.
    ///
    /// `counts` specifies the number of (scalar, base) pairs for each MSM.
    /// `bases` and `scalars` are all inputs concatenated.
    ///
    /// Returns results in Montgomery form. Convert with `.inner()` then
    /// `FujiPoint::from_mont()` for normal-form affine operations.
    pub fn msm_batch(
        counts: &[i32],
        bases: &[fuji::FujiAffine],
        scalars: &[fuji::FujiField],
    ) -> Result<Vec<fuji::FujiPoint>, fuji::FujiError> {
        fuji::msm::prl_msm_batch(counts, bases, scalars, Self::curve())
    }
}

impl<C: CurveTag> Clone for FujiAffine<C> { fn clone(&self) -> Self { *self } }
impl<C: CurveTag> Copy for FujiAffine<C> {}
impl<C: CurveTag> Clone for FujiPoint<C> { fn clone(&self) -> Self { *self } }
impl<C: CurveTag> Copy for FujiPoint<C> {}

impl<C: CurveTag> From<FujiAffine<C>> for fuji::FujiAffine { fn from(f: FujiAffine<C>) -> Self { f.0 } }
impl<C: CurveTag> From<fuji::FujiAffine> for FujiAffine<C> { fn from(f: fuji::FujiAffine) -> Self { FujiAffine(f, PhantomData) } }
impl<C: CurveTag> From<FujiPoint<C>> for fuji::FujiPoint { fn from(f: FujiPoint<C>) -> Self { f.0 } }
impl<C: CurveTag> From<fuji::FujiPoint> for FujiPoint<C> { fn from(f: fuji::FujiPoint) -> Self { FujiPoint(f, PhantomData) } }

/* ── Operator traits for FujiField<C> ─────────────────────── */

impl<C: CurveTag> Add<FujiField<C>> for FujiField<C> {
    type Output = Self;
    fn add(self, other: Self) -> Self { FujiField::add(&self, other) }
}
impl<C: CurveTag> Sub<FujiField<C>> for FujiField<C> {
    type Output = Self;
    fn sub(self, other: Self) -> Self { FujiField::sub(&self, other) }
}
impl<C: CurveTag> Mul<FujiField<C>> for FujiField<C> {
    type Output = Self;
    fn mul(self, other: Self) -> Self { FujiField::mul(&self, other) }
}
impl<C: CurveTag> Neg for FujiField<C> {
    type Output = Self;
    fn neg(self) -> Self { FujiField::neg(&self) }
}

impl<C: CurveTag> Add<&FujiField<C>> for FujiField<C> {
    type Output = Self;
    fn add(self, other: &Self) -> Self { FujiField::add(&self, *other) }
}
impl<C: CurveTag> Sub<&FujiField<C>> for FujiField<C> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self { FujiField::sub(&self, *other) }
}
impl<C: CurveTag> Mul<&FujiField<C>> for FujiField<C> {
    type Output = Self;
    fn mul(self, other: &Self) -> Self { FujiField::mul(&self, *other) }
}

impl<C: CurveTag> AddAssign for FujiField<C> {
    fn add_assign(&mut self, other: Self) { *self = FujiField::add(self, other); }
}
impl<C: CurveTag> SubAssign for FujiField<C> {
    fn sub_assign(&mut self, other: Self) { *self = FujiField::sub(self, other); }
}
impl<C: CurveTag> MulAssign for FujiField<C> {
    fn mul_assign(&mut self, other: Self) { *self = FujiField::mul(self, other); }
}
impl<C: CurveTag> AddAssign<&FujiField<C>> for FujiField<C> {
    fn add_assign(&mut self, other: &Self) { *self = FujiField::add(self, *other); }
}
impl<C: CurveTag> SubAssign<&FujiField<C>> for FujiField<C> {
    fn sub_assign(&mut self, other: &Self) { *self = FujiField::sub(self, *other); }
}
impl<C: CurveTag> MulAssign<&FujiField<C>> for FujiField<C> {
    fn mul_assign(&mut self, other: &Self) { *self = FujiField::mul(self, *other); }
}

impl<C: CurveTag> From<u64> for FujiField<C> {
    fn from(v: u64) -> Self {
        let mut bytes = [0u8; 32];
        for i in 0..8 { bytes[i] = ((v >> (i * 8)) & 0xFF) as u8; }
        Self::from_bytes(&bytes)
    }
}

impl<C: CurveTag> Sum for FujiField<C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + b)
    }
}
impl<C: CurveTag> Product for FujiField<C> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * b)
    }
}
impl<'a, C: CurveTag> Sum<&'a FujiField<C>> for FujiField<C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |a, b| a + *b)
    }
}
impl<'a, C: CurveTag> Product<&'a FujiField<C>> for FujiField<C> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |a, b| a * *b)
    }
}

/* ── subtle traits ─────────────────────────────────────────── */

use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};

impl<C: CurveTag> ConditionallySelectable for FujiField<C> {
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        let mut out = [0u8; 32];
        for i in 0..32 {
            out[i] = u8::conditional_select(&a.0.to_bytes()[i], &b.0.to_bytes()[i], choice);
        }
        Self::from_bytes(&out)
    }
}

impl<C: CurveTag> ConstantTimeEq for FujiField<C> {
    fn ct_eq(&self, other: &Self) -> Choice {
        let a = self.0.to_bytes();
        let b = other.0.to_bytes();
        let mut r = 0u8;
        for i in 0..32 { r |= a[i] ^ b[i]; }
        Choice::from((r == 0) as u8)
    }
}

/* ── ff::Field impl ────────────────────────────────────────── */

const FIELD_ZERO_INNER: fuji_sys::fuji_field = fuji_sys::fuji_field { limbs: [0u8; 32] };
const FIELD_ONE_INNER: fuji_sys::fuji_field = fuji_sys::fuji_field {
    limbs: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
};

impl<C: CurveConstants> ff::Field for FujiField<C> {
    const ZERO: Self = FujiField(fuji::FujiField(FIELD_ZERO_INNER), PhantomData);
    const ONE: Self = FujiField(fuji::FujiField(FIELD_ONE_INNER), PhantomData);

    fn random(mut rng: impl rand_core::RngCore) -> Self {
        let mut bytes = [0u8; 32];
        rng.fill_bytes(&mut bytes);
        Self::from_bytes(&bytes)
    }

    fn is_zero(&self) -> Choice { self.is_zero() }

    fn double(&self) -> Self { *self + *self }
    fn square(&self) -> Self { *self * *self }

    fn invert(&self) -> subtle::CtOption<Self> { self.invert() }
    fn sqrt(&self) -> subtle::CtOption<Self> { self.sqrt() }

    fn sqrt_ratio(num: &Self, div: &Self) -> (Choice, Self) {
        let r = *num * ff::Field::invert(div).unwrap();
        let s = ff::Field::sqrt(&r);
        (s.is_some(), s.unwrap_or(Self::ZERO))
    }
}

/* ── ff::PrimeField impl ───────────────────────────────── */

impl<C: CurveConstants> ff::PrimeField for FujiField<C> {
    type Repr = [u8; 32];

    const MODULUS: &'static str = C::MODULUS_STR;
    const NUM_BITS: u32 = 255;
    const CAPACITY: u32 = 254;
    const TWO_INV: Self = FujiField(fuji::FujiField(fuji_sys::fuji_field { limbs: C::TWO_INV_BYTES }), PhantomData);
    const S: u32 = 32;

    fn from_repr(bytes: Self::Repr) -> subtle::CtOption<Self> {
        subtle::CtOption::new(Self::from_bytes(&bytes), Choice::from(1u8))
    }

    fn to_repr(&self) -> Self::Repr { self.0.to_bytes() }

    fn is_odd(&self) -> Choice { self.is_odd() }

    const MULTIPLICATIVE_GENERATOR: Self = <Self as ff::Field>::ZERO;
    const ROOT_OF_UNITY: Self = <Self as ff::Field>::ZERO;
    const ROOT_OF_UNITY_INV: Self = <Self as ff::Field>::ZERO;
    const DELTA: Self = <Self as ff::Field>::ZERO;
}

/* ── Point misc traits ─────────────────────────────────────── */

impl<C: CurveTag> Eq for FujiPoint<C> {}
impl<C: CurveTag> PartialEq for FujiPoint<C> {
    fn eq(&self, other: &Self) -> bool {
        self.0.x_limbs() == other.0.x_limbs() &&
        !(self.is_identity() ^ other.is_identity())
    }
}
impl<C: CurveTag> std::fmt::Debug for FujiPoint<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FujiPoint({:02x?})", self.0.x_limbs())
    }
}
impl<C: CurveTag> Sum for FujiPoint<C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |a, b| a + b)
    }
}
impl<'a, C: CurveTag> Sum<&'a FujiPoint<C>> for FujiPoint<C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::identity(), |a, b| a + *b)
    }
}
impl<C: CurveScalar> Mul<FujiField<C::Scalar>> for FujiPoint<C> {
    type Output = Self;
    fn mul(self, scalar: FujiField<<C as CurveScalar>::Scalar>) -> Self {
        let bytes = scalar.0.to_bytes();
        let mut acc = FujiPoint::identity();
        let mut base = self;
        for byte in bytes.iter() {
            for bit in 0..8 {
                if byte & (1u8 << bit) != 0 { acc = acc + base; }
                base = base.double();
            }
        }
        acc
    }
}
impl<C: CurveScalar> Mul<&FujiField<C::Scalar>> for FujiPoint<C> {
    type Output = Self;
    fn mul(self, scalar: &FujiField<C::Scalar>) -> Self { self * *scalar }
}
impl<C: CurveScalar> MulAssign<FujiField<C::Scalar>> for FujiPoint<C> {
    fn mul_assign(&mut self, scalar: FujiField<C::Scalar>) { *self = *self * scalar; }
}
impl<C: CurveScalar> MulAssign<&FujiField<C::Scalar>> for FujiPoint<C> {
    fn mul_assign(&mut self, scalar: &FujiField<C::Scalar>) { *self = *self * *scalar; }
}

/* ── Point operator traits ────────────────────────────────── */

impl<C: CurveTag> Add<FujiPoint<C>> for FujiPoint<C> {
    type Output = Self;
    fn add(self, other: Self) -> Self { FujiPoint::add(&self, &other) }
}
impl<C: CurveTag> Sub<FujiPoint<C>> for FujiPoint<C> {
    type Output = Self;
    fn sub(self, other: Self) -> Self { self.add(other.neg()) }
}
impl<C: CurveTag> Neg for FujiPoint<C> {
    type Output = Self;
    fn neg(self) -> Self { FujiPoint::neg(&self) }
}
impl<C: CurveTag> Add<&FujiPoint<C>> for FujiPoint<C> {
    type Output = Self;
    fn add(self, other: &Self) -> Self { FujiPoint::add(&self, other) }
}
impl<C: CurveTag> Sub<&FujiPoint<C>> for FujiPoint<C> {
    type Output = Self;
    fn sub(self, other: &Self) -> Self { self.add(other.neg()) }
}
impl<C: CurveTag> AddAssign for FujiPoint<C> {
    fn add_assign(&mut self, other: Self) { *self = FujiPoint::add(self, &other); }
}
impl<C: CurveTag> SubAssign for FujiPoint<C> {
    fn sub_assign(&mut self, other: Self) { *self = self.add(other.neg()); }
}
impl<C: CurveTag> AddAssign<&FujiPoint<C>> for FujiPoint<C> {
    fn add_assign(&mut self, other: &Self) { *self = FujiPoint::add(self, other); }
}
impl<C: CurveTag> SubAssign<&FujiPoint<C>> for FujiPoint<C> {
    fn sub_assign(&mut self, other: &Self) { *self = self.add(other.neg()); }
}

/* ── group::Group impl ──────────────────────────────────── */

impl<C: CurveConstants> group::Group for FujiPoint<C> where C::Scalar: CurveConstants {
    type Scalar = FujiField<C::Scalar>;

    fn random(mut rng: impl rand_core::RngCore) -> Self {
        let g = Self::generator();
        let s = <FujiField<C::Scalar> as ff::Field>::random(&mut rng);
        g * s
    }

    fn identity() -> Self { Self::identity() }

    fn generator() -> Self {
        let aff = if C::curve_id() == 0 { FujiAffine::gen_pallas() } else { FujiAffine::gen_vesta() };
        Self::from_affine(&aff)
    }

    fn is_identity(&self) -> Choice { Choice::from(self.is_identity() as u8) }

    fn double(&self) -> Self { Self::double(self) }
}

/* ── pasta_curves From impls ─────────────────────────────── */

use ff::PrimeField;

impl From<FujiField<PallasTag>> for pasta_curves::Fp {
    fn from(f: FujiField<PallasTag>) -> Self {
        pasta_curves::Fp::from_repr(f.0.to_bytes()).unwrap()
    }
}
impl From<pasta_curves::Fp> for FujiField<PallasTag> {
    fn from(f: pasta_curves::Fp) -> Self {
        Self::from_bytes(&f.to_repr())
    }
}

impl From<FujiField<VestaTag>> for pasta_curves::Fq {
    fn from(f: FujiField<VestaTag>) -> Self {
        pasta_curves::Fq::from_repr(f.0.to_bytes()).unwrap()
    }
}
impl From<pasta_curves::Fq> for FujiField<VestaTag> {
    fn from(f: pasta_curves::Fq) -> Self {
        Self::from_bytes(&f.to_repr())
    }
}


