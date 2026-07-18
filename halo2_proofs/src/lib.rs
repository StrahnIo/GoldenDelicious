//! # halo2_proofs
//!
//! ## Feature flags
//!
//! - **`fuji`** — Enables Apple Silicon AMX/SME acceleration via the [`fuji`] library.
//!   Requires Apple Silicon (M1–M4) and `DYLD_LIBRARY_PATH` pointing to
//!   `libfuji.dylib`. Uses AMX on M1–M4 and additionally detects SME on M4+.
//!   The `fuji` feature performs no runtime processor detection;
//!   it is the caller's responsibility to ensure the target is a compatible
//!   Apple Silicon system.

#![cfg_attr(docsrs, feature(doc_cfg))]
// The actual lints we want to disable.
#![allow(clippy::op_ref, clippy::many_single_char_names)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]

pub mod arithmetic;
pub mod circuit;
pub use pasta_curves as pasta;
mod multicore;
pub mod plonk;
pub mod poly;
pub mod transcript;

pub mod dev;
mod helpers;
