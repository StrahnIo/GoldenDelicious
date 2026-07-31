//! Safe Rust wrappers for `libfuji` — AMX-accelerated elliptic curve arithmetic
//! for Halo2 proving on Apple Silicon.
#![allow(rustdoc::broken_intra_doc_links)]
//!
//! # Quick start
//!
//! ```ignore
//! use fuji::*;
//!
//! let g = FujiAffine::gen_pallas();
//! let one = FujiField::one();
//! let two = FujiField::from_bytes(&[2u8; 32]);
//!
//! // Single MSM: 1·G + 2·G = 3·G
//! let result = msm::msm_eval(&[g, g], &[one, two], FujiCurve::Pallas).unwrap();
//! assert!(!result.is_identity());
//! ```
//!
//! # Architecture
//!
//! | Module | Contents |
//! |---|---|
//! | [`field`] | [`FujiField`] — 255-bit Pasta field element arithmetic |
//! | [`curve`] | [`FujiAffine`], [`FujiPoint`] — Pallas/Vesta curve operations |
//! | [`msm`] | Multi-scalar multiplication (Pippenger's bucket method) |
//! | [`detection`] | AMX availability and CPU information |
//! | [`error`] | [`FujiError`] — typed error handling |
//!
//! # Low-level access
//!
//! For direct FFI access see the [`fuji-sys`] crate.

pub mod error;
pub mod detection;
pub mod field;
pub mod curve;
pub mod msm;
pub mod ntt;
pub mod sme;
pub mod prl;
pub mod batch_field;
pub mod eval;
pub mod srs;
pub use srs::SrsContext;

pub use error::FujiError;
pub use field::FujiField;
pub use curve::{FujiAffine, FujiPoint};

/// Restores signal handlers and frees library-global state.
///
/// Safe to call multiple times; idempotent. Call before process exit
/// to avoid spurious SIGSEGV from lingering AMX-detection signal handlers.
/// The C library also registers a `__attribute__((destructor))` for automatic
/// cleanup, but explicit invocation is preferred in Rust binaries.
pub fn cleanup() {
    unsafe { fuji_sys::fuji_cleanup_ffi(); }
}

/// Pasta curve selector.
///
/// Both curves share the equation `y² = x³ + 5` and form a 2-cycle:
/// the base field of one is the scalar field of the other.
///
/// - [`Pallas`](FujiCurve::Pallas): base field `0x4000...0001`
/// - [`Vesta`](FujiCurve::Vesta): base field `0x4000...0001` (different low bytes)
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum FujiCurve {
    Pallas = 0,
    Vesta = 1,
}
