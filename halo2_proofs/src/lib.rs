//! # halo2_proofs
//!
//! ## Feature flags
//!
//! - **`fuji`** — Enables Apple Silicon SME detection via the [`fuji`] library.
//!   Reserved for future SME-accelerated MSM dispatch. Field arithmetic and
//!   multi-scalar multiplication use Rust NEON Montgomery (`pasta_curves`).
//!   Requires Apple Silicon (M4+) and `DYLD_LIBRARY_PATH` pointing to
//!   `libfuji.dylib`.

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
