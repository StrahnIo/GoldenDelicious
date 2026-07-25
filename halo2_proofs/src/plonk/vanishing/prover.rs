use std::iter;

use ff::Field;
use group::Curve;
use rand_core::RngCore;

use super::Argument;
use crate::{
    arithmetic::{eval_polynomial, CurveAffine},
    plonk::{ChallengeX, ChallengeY, Error},
    poly::{
        self,
        commitment::{Blind, Params},
        multiopen::ProverQuery,
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, Polynomial,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
};

pub(in crate::plonk) struct Committed<C: CurveAffine> {
    random_poly: Polynomial<C::Scalar, Coeff>,
    random_blind: Blind<C::Scalar>,
}

pub(in crate::plonk) struct Constructed<C: CurveAffine> {
    h_pieces: Vec<Polynomial<C::Scalar, Coeff>>,
    h_blinds: Vec<Blind<C::Scalar>>,
    committed: Committed<C>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    h_poly: Polynomial<C::Scalar, Coeff>,
    h_blind: Blind<C::Scalar>,
    committed: Committed<C>,
}

impl<C: CurveAffine> Argument<C> {
    pub(in crate::plonk) fn commit<E: EncodedChallenge<C>, R: RngCore, T: TranscriptWrite<C, E>>(
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        // Sample a random polynomial of degree n - 1
        let mut random_poly = domain.empty_coeff();
        for coeff in random_poly.iter_mut() {
            *coeff = C::Scalar::random(&mut rng);
        }
        // Sample a random blinding factor
        let random_blind = Blind(C::Scalar::random(rng));

        // Commit
        let c = params.commit(&random_poly, random_blind).to_affine();
        transcript.write_point(c)?;

        Ok(Committed {
            random_poly,
            random_blind,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::plonk) fn construct<
        E: EncodedChallenge<C>,
        Ev: Copy + Send + Sync,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        params: &Params<C>,
        domain: &EvaluationDomain<C::Scalar>,
        evaluator: poly::Evaluator<Ev, C::Scalar, ExtendedLagrangeCoeff>,
        expressions: impl Iterator<Item = poly::Ast<Ev, C::Scalar, ExtendedLagrangeCoeff>>,
        y: ChallengeY<C>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Constructed<C>, Error> {
        let _t0 = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let h_poly = poly::Ast::distribute_powers(expressions, *y);
        if let Some(ref t0) = _t0 { eprintln!("[perf]   distribute_powers: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0); }

        let _t1 = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let h_poly = evaluator.evaluate(&h_poly, domain);
        if let Some(ref t1) = _t1 { eprintln!("[perf]   evaluator_evaluate: {:.1}ms", t1.elapsed().as_secs_f64() * 1000.0); }

        let _t2 = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let h_poly = domain.divide_by_vanishing_poly(h_poly);
        if let Some(ref t2) = _t2 { eprintln!("[perf]   divide_by_vanishing_poly: {:.1}ms", t2.elapsed().as_secs_f64() * 1000.0); }

        let _t3 = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let h_poly = domain.extended_to_coeff(h_poly);
        if let Some(ref t3) = _t3 { eprintln!("[perf]   extended_to_coeff: {:.1}ms", t3.elapsed().as_secs_f64() * 1000.0); }

        let _t4 = if std::env::var("PERF_DEBUG").is_ok() { Some(std::time::Instant::now()) } else { None };
        let h_pieces = h_poly
            .chunks_exact(params.n as usize)
            .map(|v| domain.coeff_from_vec(v.to_vec()))
            .collect::<Vec<_>>();
        drop(h_poly);
        let h_blinds: Vec<_> = h_pieces
            .iter()
            .map(|_| Blind(C::Scalar::random(&mut rng)))
            .collect();

        let h_refs: Vec<&Polynomial<C::Scalar, Coeff>> = h_pieces.iter().collect();
        let h_commitments_projective = params.commit_batch(&h_refs, &h_blinds);
        let mut h_commitments = vec![C::identity(); h_commitments_projective.len()];
        C::Curve::batch_normalize(&h_commitments_projective, &mut h_commitments);
        let h_commitments = h_commitments;
        if let Some(ref t4) = _t4 { eprintln!("[perf]   h_split_blind_commit: {:.1}ms", t4.elapsed().as_secs_f64() * 1000.0); }

        for c in h_commitments.iter() {
            transcript.write_point(*c)?;
        }

        Ok(Constructed {
            h_pieces,
            h_blinds,
            committed: self,
        })
    }
}

impl<C: CurveAffine> Constructed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        x: ChallengeX<C>,
        xn: C::Scalar,
        domain: &EvaluationDomain<C::Scalar>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let h_poly = self
            .h_pieces
            .iter()
            .rev()
            .fold(domain.empty_coeff(), |acc, eval| acc * xn + eval);

        let h_blind = self
            .h_blinds
            .iter()
            .rev()
            .fold(Blind(C::Scalar::ZERO), |acc, eval| acc * Blind(xn) + *eval);

        let random_eval = eval_polynomial(&self.committed.random_poly, *x);
        transcript.write_scalar(random_eval)?;

        Ok(Evaluated {
            h_poly,
            h_blind,
            committed: self.committed,
        })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open(
        &self,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'_, C>> + Clone {
        iter::empty()
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.h_poly,
                blind: self.h_blind,
            }))
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.committed.random_poly,
                blind: self.committed.random_blind,
            }))
    }
}
