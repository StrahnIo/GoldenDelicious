use super::circuit::{Any, Column};
use crate::{
    arithmetic::CurveAffine,
    poly::{Coeff, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial},
};
use ff::PrimeField;
use std::io::{Read, Write};

pub(crate) mod keygen;
pub(crate) mod prover;
pub(crate) mod verifier;

/// A permutation argument.
#[derive(Debug, Clone)]
pub(crate) struct Argument {
    /// A sequence of columns involved in the argument.
    columns: Vec<Column<Any>>,
}

impl Argument {
    pub(crate) fn new() -> Self {
        Argument { columns: vec![] }
    }

    /// Returns the minimum circuit degree required by the permutation argument.
    /// The argument may use larger degree gates depending on the actual
    /// circuit's degree and how many columns are involved in the permutation.
    pub(crate) fn required_degree(&self) -> usize {
        // degree 2:
        // l_0(X) * (1 - z(X)) = 0
        //
        // We will fit as many polynomials p_i(X) as possible
        // into the required degree of the circuit, so the
        // following will not affect the required degree of
        // this middleware.
        //
        // (1 - (l_last(X) + l_blind(X))) * (
        //   z(\omega X) \prod (p(X) + \beta s_i(X) + \gamma)
        // - z(X) \prod (p(X) + \delta^i \beta X + \gamma)
        // )
        //
        // On the first sets of columns, except the first
        // set, we will do
        //
        // l_0(X) * (z(X) - z'(\omega^(last) X)) = 0
        //
        // where z'(X) is the permutation for the previous set
        // of columns.
        //
        // On the final set of columns, we will do
        //
        // degree 3:
        // l_last(X) * (z'(X)^2 - z'(X)) = 0
        //
        // which will allow the last value to be zero to
        // ensure the argument is perfectly complete.

        // There are constraints of degree 3 regardless of the
        // number of columns involved.
        3
    }

    pub(crate) fn add_column(&mut self, column: Column<Any>) {
        if !self.columns.contains(&column) {
            self.columns.push(column);
        }
    }

    pub(crate) fn get_columns(&self) -> Vec<Column<Any>> {
        self.columns.clone()
    }

    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&(self.columns.len() as u64).to_le_bytes())?;
        for c in &self.columns {
            Column::<Any>::write_column(c, writer)?;
        }
        Ok(())
    }
    pub(crate) fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        let n = u64::from_le_bytes(buf) as usize;
        let mut columns = Vec::with_capacity(n);
        for _ in 0..n {
            columns.push(Column::<Any>::read_column(reader)?);
        }
        Ok(Argument { columns })
    }
}

/// The verifying key for a single permutation argument.
#[derive(Clone, Debug)]
pub(crate) struct VerifyingKey<C: CurveAffine> {
    commitments: Vec<C>,
}

impl<C: CurveAffine> VerifyingKey<C> where C::Scalar: PrimeField {
    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
        writer.write_all(&(self.commitments.len() as u64).to_le_bytes())?;
        for c in &self.commitments {
            let bytes = c.to_bytes();
            writer.write_all(bytes.as_ref())?;
        }
        Ok(())
    }
    pub(crate) fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        let n = u64::from_le_bytes(buf) as usize;
        let mut commitments = Vec::with_capacity(n);
        for _ in 0..n {
            let mut repr = C::Repr::default();
            reader.read_exact(repr.as_mut())?;
            let c = C::from_bytes(&repr).unwrap_or_else(C::identity);
            commitments.push(c);
        }
        Ok(VerifyingKey { commitments })
    }
}

/// The proving key for a single permutation argument.
#[derive(Clone, Debug)]
pub(crate) struct ProvingKey<C: CurveAffine> {
    permutations: Vec<Polynomial<C::Scalar, LagrangeCoeff>>,
    polys: Vec<Polynomial<C::Scalar, Coeff>>,
    pub(super) cosets: Vec<Polynomial<C::Scalar, ExtendedLagrangeCoeff>>,
}

impl<C: CurveAffine> ProvingKey<C> where C::Scalar: PrimeField {
    pub(crate) fn write<W: Write>(&self, writer: &mut W) -> std::io::Result<()> {
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
        wv(&self.permutations, writer)?;
        wv2(&self.polys, writer)?;
        wv3(&self.cosets, writer)?;
        Ok(())
    }
    pub(crate) fn read<R: Read>(reader: &mut R) -> std::io::Result<Self> {
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
            permutations: rv(reader)?,
            polys: rv2(reader)?,
            cosets: rv3(reader)?,
        })
    }
}
