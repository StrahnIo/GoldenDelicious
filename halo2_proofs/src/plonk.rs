//! This module provides an implementation of a variant of (Turbo)[PLONK][plonk]
//! that is designed specifically for the polynomial commitment scheme described
//! in the [Halo][halo] paper.
//!
//! [halo]: https://eprint.iacr.org/2019/1021
//! [plonk]: https://eprint.iacr.org/2019/953

use blake2b_simd::Params as Blake2bParams;
use group::ff::{Field, FromUniformBytes, PrimeField};
use std::io::{Read, Write};

use crate::arithmetic::CurveAffine;
use crate::poly::{
    Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, PinnedEvaluationDomain,
    Polynomial,
};
use crate::transcript::{ChallengeScalar, EncodedChallenge, Transcript};

mod assigned;
mod circuit;
mod error;
mod keygen;
mod lookup;
pub(crate) mod permutation;
mod vanishing;

mod prover;
mod verifier;

pub use assigned::*;
pub use circuit::*;
pub use error::*;
pub use keygen::*;
pub use prover::*;
pub use verifier::*;

use std::io;

/// This is a verifying key which allows for the verification of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct VerifyingKey<C: CurveAffine> {
    domain: EvaluationDomain<C::Scalar>,
    fixed_commitments: Vec<C>,
    permutation: permutation::VerifyingKey<C>,
    cs: ConstraintSystem<C::Scalar>,
    /// Cached maximum degree of `cs` (which doesn't change after construction).
    cs_degree: usize,
    /// The representative of this `VerifyingKey` in transcripts.
    transcript_repr: C::Scalar,
}

impl<C: CurveAffine> VerifyingKey<C>
where
    C::Scalar: FromUniformBytes<64>,
{
    fn from_parts(
        domain: EvaluationDomain<C::Scalar>,
        fixed_commitments: Vec<C>,
        permutation: permutation::VerifyingKey<C>,
        cs: ConstraintSystem<C::Scalar>,
    ) -> Self {
        // Compute cached values.
        let cs_degree = cs.degree();

        let mut vk = Self {
            domain,
            fixed_commitments,
            permutation,
            cs,
            cs_degree,
            // Temporary, this is not pinned.
            transcript_repr: C::Scalar::ZERO,
        };

        let mut hasher = Blake2bParams::new()
            .hash_length(64)
            .personal(b"Halo2-Verify-Key")
            .to_state();

        let s = format!("{:?}", vk.pinned());

        hasher.update(&(s.len() as u64).to_le_bytes());
        hasher.update(s.as_bytes());

        // Hash in final Blake2bState
        vk.transcript_repr = C::Scalar::from_uniform_bytes(hasher.finalize().as_array());

        vk
    }
}

impl<C: CurveAffine> VerifyingKey<C> {
    /// Hashes a verification key into a transcript.
    pub fn hash_into<E: EncodedChallenge<C>, T: Transcript<C, E>>(
        &self,
        transcript: &mut T,
    ) -> io::Result<()> {
        transcript.common_scalar(self.transcript_repr)?;

        Ok(())
    }

    /// Obtains a pinned representation of this verification key that contains
    /// the minimal information necessary to reconstruct the verification key.
    pub fn pinned(&self) -> PinnedVerificationKey<'_, C> {
        PinnedVerificationKey {
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            domain: self.domain.pinned(),
            fixed_commitments: &self.fixed_commitments,
            permutation: &self.permutation,
            cs: self.cs.pinned(),
        }
    }
}

/// Minimal representation of a verification key that can be used to identify
/// its active contents.
#[allow(dead_code)]
#[derive(Debug)]
pub struct PinnedVerificationKey<'a, C: CurveAffine> {
    base_modulus: &'static str,
    scalar_modulus: &'static str,
    domain: PinnedEvaluationDomain<'a, C::Scalar>,
    cs: PinnedConstraintSystem<'a, C::Scalar>,
    fixed_commitments: &'a Vec<C>,
    permutation: &'a permutation::VerifyingKey<C>,
}
/// This is a proving key which allows for the creation of proofs for a
/// particular circuit.
#[derive(Clone, Debug)]
pub struct ProvingKey<C: CurveAffine> {
    vk: VerifyingKey<C>,
    l0: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_blind: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    l_last: Polynomial<C::Scalar, ExtendedLagrangeCoeff>,
    fixed_values: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    fixed_polys: Vec<Polynomial<C::Scalar, Coeff>>,
    fixed_cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
    permutation: permutation::ProvingKey<C>,
}

impl<C: CurveAffine> ProvingKey<C> where C::Scalar: PrimeField {
    /// Get the underlying [`VerifyingKey`].
    pub fn get_vk(&self) -> &VerifyingKey<C> {
        &self.vk
    }

    /// Serialize this proving key to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.vk.write(writer)?;
        self.l0.write(writer)?;
        self.l_blind.write(writer)?;
        self.l_last.write(writer)?;
        let wv = |v: &[Polynomial<C::Scalar, LagrangeCoeff>], w: &mut W| -> std::io::Result<()> {
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for p in v { p.write(w)?; }
            Ok(())
        };
        let wv2 = |v: &[Polynomial<C::Scalar, Coeff>], w: &mut W| -> std::io::Result<()> {
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for p in v { p.write(w)?; }
            Ok(())
        };
        let wv3 = |v: &[Polynomial<C::Scalar, ExtendedLagrangeCoeff>], w: &mut W| -> std::io::Result<()> {
            w.write_all(&(v.len() as u64).to_le_bytes())?;
            for p in v { p.write(w)?; }
            Ok(())
        };
        wv(&self.fixed_values, writer)?;
        wv2(&self.fixed_polys, writer)?;
        wv3(&self.fixed_cosets, writer)?;
        self.permutation.write(writer)?;
        Ok(())
    }

    /// Deserialize a proving key from a reader.
    pub fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let rv = |r: &mut R| -> std::io::Result<Vec<Polynomial<C::Scalar, LagrangeCoeff>>> {
            let mut buf = [0u8; 8]; r.read_exact(&mut buf)?;
            let n = u64::from_le_bytes(buf) as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n { v.push(Polynomial::read(r)?); }
            Ok(v)
        };
        let rv2 = |r: &mut R| -> std::io::Result<Vec<Polynomial<C::Scalar, Coeff>>> {
            let mut buf = [0u8; 8]; r.read_exact(&mut buf)?;
            let n = u64::from_le_bytes(buf) as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n { v.push(Polynomial::read(r)?); }
            Ok(v)
        };
        let rv3 = |r: &mut R| -> std::io::Result<Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>> {
            let mut buf = [0u8; 8]; r.read_exact(&mut buf)?;
            let n = u64::from_le_bytes(buf) as usize;
            let mut v = Vec::with_capacity(n);
            for _ in 0..n { v.push(Polynomial::read(r)?); }
            Ok(v)
        };
        Ok(ProvingKey {
            vk: VerifyingKey::read(reader)?,
            l0: Polynomial::read(reader)?,
            l_blind: Polynomial::read(reader)?,
            l_last: Polynomial::read(reader)?,
            fixed_values: rv(reader)?,
            fixed_polys: rv2(reader)?,
            fixed_cosets: rv3(reader)?,
            permutation: permutation::ProvingKey::read(reader)?,
        })
    }
}

impl<C: CurveAffine> VerifyingKey<C> where C::Scalar: PrimeField {
    /// Get the underlying [`EvaluationDomain`].
    pub fn get_domain(&self) -> &EvaluationDomain<C::Scalar> {
        &self.domain
    }

    /// Serialize this verifying key to a writer.
    pub fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        self.domain.write(writer)?;
        writer.write_all(&(self.fixed_commitments.len() as u64).to_le_bytes())?;
        for c in &self.fixed_commitments {
            let bytes = c.to_bytes();
            writer.write_all(bytes.as_ref())?;
        }
        self.permutation.write(writer)?;
        self.cs.write(writer)?;
        writer.write_all(&(self.cs_degree as u64).to_le_bytes())?;
        Ok(())
    }

    /// Deserialize a verifying key from a reader.
    pub fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let domain = EvaluationDomain::read(reader)?;
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        let nfc = u64::from_le_bytes(buf) as usize;
        let mut fixed_commitments = Vec::with_capacity(nfc);
        for _ in 0..nfc {
            let mut repr = C::Repr::default();
            reader.read_exact(repr.as_mut())?;
            fixed_commitments.push(C::from_bytes(&repr).unwrap_or_else(C::identity));
        }
        let permutation = permutation::VerifyingKey::read(reader)?;
        let cs = ConstraintSystem::read(reader)?;
        reader.read_exact(&mut buf)?;
        let cs_degree = u64::from_le_bytes(buf) as usize;
        let transcript_repr = C::Scalar::ZERO;
        Ok(VerifyingKey { domain, fixed_commitments, permutation, cs, cs_degree, transcript_repr })
    }
}

#[derive(Clone, Copy, Debug)]
struct Theta;
type ChallengeTheta<F> = ChallengeScalar<F, Theta>;

#[derive(Clone, Copy, Debug)]
struct Beta;
type ChallengeBeta<F> = ChallengeScalar<F, Beta>;

#[derive(Clone, Copy, Debug)]
struct Gamma;
type ChallengeGamma<F> = ChallengeScalar<F, Gamma>;

#[derive(Clone, Copy, Debug)]
struct Y;
type ChallengeY<F> = ChallengeScalar<F, Y>;

#[derive(Clone, Copy, Debug)]
struct X;
type ChallengeX<F> = ChallengeScalar<F, X>;
