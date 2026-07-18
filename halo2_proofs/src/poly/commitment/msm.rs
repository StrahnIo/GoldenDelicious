use super::Params;
use crate::arithmetic::{best_multiexp, CurveAffine};
use ff::{Field, PrimeField};
use group::Group;

#[cfg(feature = "fuji")]
use fuji::{FujiAffine, FujiField, FujiCurve};
use std::collections::BTreeMap;

/// A multiscalar multiplication in the polynomial commitment scheme
#[derive(Debug, Clone)]
pub struct MSM<'a, C: CurveAffine> {
    pub(crate) params: &'a Params<C>,
    g_scalars: Option<Vec<C::Scalar>>,
    w_scalar: Option<C::Scalar>,
    u_scalar: Option<C::Scalar>,
    other: BTreeMap<C::Base, (C::Scalar, C::Base)>,
    #[cfg(feature = "fuji")]
    fuji_scalars: Vec<C::Scalar>,
    #[cfg(feature = "fuji")]
    fuji_bases_mont: Vec<FujiAffine>,
    #[cfg(feature = "fuji")]
    fuji_g_scalars: Vec<C::Scalar>,
    #[cfg(feature = "fuji")]
    fuji_g_bases: Vec<FujiAffine>,
    #[cfg(feature = "fuji")]
    fuji_w_scalar: Option<C::Scalar>,
    #[cfg(feature = "fuji")]
    fuji_w_base: FujiAffine,
    #[cfg(feature = "fuji")]
    fuji_u_scalar: Option<C::Scalar>,
    #[cfg(feature = "fuji")]
    fuji_u_base: FujiAffine,
}

impl<'a, C: CurveAffine> MSM<'a, C> {
    /// Create a new, empty MSM using the provided parameters.
    pub fn new(params: &'a Params<C>) -> Self {
        let g_scalars = None;
        let w_scalar = None;
        let u_scalar = None;
        let other = BTreeMap::new();

        MSM {
            params,
            g_scalars,
            w_scalar,
            u_scalar,
            other,
            #[cfg(feature = "fuji")]
            fuji_scalars: Vec::new(),
            #[cfg(feature = "fuji")]
            fuji_bases_mont: Vec::new(),
            #[cfg(feature = "fuji")]
            fuji_g_scalars: Vec::new(),
            #[cfg(feature = "fuji")]
            fuji_g_bases: {
                let curve = FujiCurve::Pallas;
                params.g.iter().map(|p| point_to_fuji_affine_mont(p, curve)).collect()
            },
            #[cfg(feature = "fuji")]
            fuji_w_scalar: None,
            #[cfg(feature = "fuji")]
            fuji_w_base: point_to_fuji_affine_mont(&params.w, FujiCurve::Pallas),
            #[cfg(feature = "fuji")]
            fuji_u_scalar: None,
            #[cfg(feature = "fuji")]
            fuji_u_base: point_to_fuji_affine_mont(&params.u, FujiCurve::Pallas),
        }
    }

    /// Add another multiexp into this one
    pub fn add_msm(&mut self, other: &Self) {
        for (x, (scalar, y)) in other.other.iter() {
            self.other
                .entry(*x)
                .and_modify(|(our_scalar, our_y)| {
                    if our_y == y {
                        *our_scalar += *scalar;
                    } else {
                        assert!(*our_y == -*y);
                        *our_scalar -= *scalar;
                    }
                })
                .or_insert((*scalar, *y));
        }

        if let Some(g_scalars) = &other.g_scalars {
            self.add_to_g_scalars(g_scalars);
        }

        if let Some(w_scalar) = &other.w_scalar {
            self.add_to_w_scalar(*w_scalar);
        }

        if let Some(u_scalar) = &other.u_scalar {
            self.add_to_u_scalar(*u_scalar);
        }

        #[cfg(feature = "fuji")]
        {
            self.fuji_scalars.extend(other.fuji_scalars.iter());
            self.fuji_bases_mont.extend(other.fuji_bases_mont.iter());
            self.fuji_g_scalars.extend(other.fuji_g_scalars.iter());
            if let Some(ws) = other.fuji_w_scalar {
                self.fuji_w_scalar = self.fuji_w_scalar.map_or(Some(ws), |a| Some(a + &ws));
            }
            if let Some(us) = other.fuji_u_scalar {
                self.fuji_u_scalar = self.fuji_u_scalar.map_or(Some(us), |a| Some(a + &us));
            }
        }
    }

    /// Add arbitrary term (the scalar and the point)
    pub fn append_term(&mut self, scalar: C::Scalar, point: C) {
        if !bool::from(point.is_identity()) {
            let xy = point.coordinates().unwrap();
            let x = *xy.x();
            let y = *xy.y();

            self.other
                .entry(x)
                .and_modify(|(our_scalar, our_y)| {
                    if *our_y == y {
                        *our_scalar += scalar;
                    } else {
                        assert!(*our_y == -y);
                        *our_scalar -= scalar;
                    }
                })
                .or_insert((scalar, y));

            #[cfg(feature = "fuji")]
            {
                use ff::PrimeField;
                self.fuji_scalars.push(scalar);
                let x_bytes = {
                    let repr = x.to_repr();
                    let mut buf = [0u8; 32];
                    buf.copy_from_slice(repr.as_ref());
                    buf
                };
                let y_bytes = {
                    let repr = y.to_repr();
                    let mut buf = [0u8; 32];
                    buf.copy_from_slice(repr.as_ref());
                    buf
                };
                self.fuji_bases_mont.push(FujiAffine::from_coordinates(
                    FujiField::from_bytes(&x_bytes).to_mont(FujiCurve::Pallas),
                    FujiField::from_bytes(&y_bytes).to_mont(FujiCurve::Pallas),
                ));
            }
        }
    }

    /// Add a value to the first entry of `g_scalars`.
    pub fn add_constant_term(&mut self, constant: C::Scalar) {
        if let Some(g_scalars) = self.g_scalars.as_mut() {
            g_scalars[0] += &constant;
        } else {
            let mut g_scalars = vec![C::Scalar::ZERO; self.params.n as usize];
            g_scalars[0] += &constant;
            self.g_scalars = Some(g_scalars);
        }
        #[cfg(feature = "fuji")]
        {
            let n = self.params.n as usize;
            if self.fuji_g_scalars.is_empty() {
                self.fuji_g_scalars = vec![C::Scalar::ZERO; n];
            }
            self.fuji_g_scalars[0] = self.fuji_g_scalars[0] + constant;
        }
    }

    /// Add a vector of scalars to `g_scalars`. This function will panic if the
    /// caller provides a slice of scalars that is not of length `params.n`.
    pub fn add_to_g_scalars(&mut self, scalars: &[C::Scalar]) {
        assert_eq!(scalars.len(), self.params.n as usize);
        if let Some(g_scalars) = &mut self.g_scalars {
            for (g_scalar, scalar) in g_scalars.iter_mut().zip(scalars.iter()) {
                *g_scalar += scalar;
            }
        } else {
            self.g_scalars = Some(scalars.to_vec());
        }
        #[cfg(feature = "fuji")]
        {
            if self.fuji_g_scalars.is_empty() {
                self.fuji_g_scalars = scalars.to_vec();
            } else {
                for (a, b) in self.fuji_g_scalars.iter_mut().zip(scalars.iter()) {
                    *a = *a + *b;
                }
            }
        }
    }

    /// Add to `w_scalar`
    pub fn add_to_w_scalar(&mut self, scalar: C::Scalar) {
        self.w_scalar = self.w_scalar.map_or(Some(scalar), |a| Some(a + &scalar));
        #[cfg(feature = "fuji")] {
            self.fuji_w_scalar = self.fuji_w_scalar.map_or(Some(scalar), |a| Some(a + &scalar));
        }
    }

    /// Add to `u_scalar`
    pub fn add_to_u_scalar(&mut self, scalar: C::Scalar) {
        self.u_scalar = self.u_scalar.map_or(Some(scalar), |a| Some(a + &scalar));
        #[cfg(feature = "fuji")] {
            self.fuji_u_scalar = self.fuji_u_scalar.map_or(Some(scalar), |a| Some(a + &scalar));
        }
    }

    /// Scale all scalars in the MSM by some scaling factor
    pub fn scale(&mut self, factor: C::Scalar) {
        if let Some(g_scalars) = &mut self.g_scalars {
            for g_scalar in g_scalars {
                *g_scalar *= &factor;
            }
        }

        for other in self.other.values_mut() {
            other.0 *= factor;
        }

        self.w_scalar = self.w_scalar.map(|a| a * &factor);
        self.u_scalar = self.u_scalar.map(|a| a * &factor);

        #[cfg(feature = "fuji")]
        {
            for gs in self.fuji_g_scalars.iter_mut() {
                *gs = *gs * factor;
            }
            self.fuji_w_scalar = self.fuji_w_scalar.map(|a| a * factor);
            self.fuji_u_scalar = self.fuji_u_scalar.map(|a| a * factor);
        }
    }

    /// Perform multiexp and check that it results in zero
    pub fn eval(self) -> bool
    where
        C::Scalar: ff::PrimeField,
        C::Base: ff::PrimeField,
    {
        let len = self.g_scalars.as_ref().map(|v| v.len()).unwrap_or(0)
            + self.w_scalar.map(|_| 1).unwrap_or(0)
            + self.u_scalar.map(|_| 1).unwrap_or(0)
            + self.other.len();
        let mut scalars: Vec<C::Scalar> = Vec::with_capacity(len);
        let mut bases: Vec<C> = Vec::with_capacity(len);

        scalars.extend(self.other.values().map(|(scalar, _)| scalar));
        bases.extend(
            self.other
                .iter()
                .map(|(x, (_, y))| C::from_xy(*x, *y).unwrap()),
        );

        if let Some(w_scalar) = self.w_scalar {
            scalars.push(w_scalar);
            bases.push(self.params.w);
        }

        if let Some(u_scalar) = self.u_scalar {
            scalars.push(u_scalar);
            bases.push(self.params.u);
        }

        if let Some(g_scalars) = &self.g_scalars {
            scalars.extend(g_scalars);
            bases.extend(self.params.g.iter());
        }

        assert_eq!(scalars.len(), len);

        #[cfg(feature = "fuji")]
        {
            use crate::arithmetic::fuji;
            if fuji::fuji_available() {
                let total = self.fuji_scalars.len()
                    + self.fuji_g_scalars.len()
                    + self.fuji_w_scalar.map(|_| 1).unwrap_or(0)
                    + self.fuji_u_scalar.map(|_| 1).unwrap_or(0);
                if total >= 256 {
                    let mut fs: Vec<C::Scalar> = Vec::with_capacity(total);
                    let mut fb: Vec<FujiAffine> = Vec::with_capacity(total);
                    fs.extend(self.fuji_scalars.iter());
                    fb.extend(self.fuji_bases_mont.iter());
                    if !self.fuji_g_scalars.is_empty() {
                        fs.extend(self.fuji_g_scalars.iter());
                        fb.extend(self.fuji_g_bases.iter());
                    }
                    if let Some(ws) = self.fuji_w_scalar {
                        fs.push(ws);
                        fb.push(self.fuji_w_base);
                    }
                    if let Some(us) = self.fuji_u_scalar {
                        fs.push(us);
                        fb.push(self.fuji_u_base);
                    }
                    if let Some(result) = fuji::try_multiexp_precomputed::<C>(&fs, &fb) {
                        return bool::from(result.is_identity());
                    }
                } else if self.params.n >= 64 {
                    if let Some(result) = fuji::try_multiexp::<C>(&scalars, &bases) {
                        return bool::from(result.is_identity());
                    }
                }
            }
        }

        bool::from(best_multiexp(&scalars, &bases).is_identity())
    }
}

#[cfg(feature = "fuji")]
fn point_to_fuji_affine_mont<C: CurveAffine>(p: &C, curve: FujiCurve) -> FujiAffine
where
    C::Base: ff::PrimeField,
{
    let coords = p.coordinates().unwrap();
    let x_repr = coords.x().to_repr();
    let y_repr = coords.y().to_repr();
    let mut xb = [0u8; 32]; xb.copy_from_slice(x_repr.as_ref());
    let mut yb = [0u8; 32]; yb.copy_from_slice(y_repr.as_ref());
    FujiAffine::from_coordinates(
        FujiField::from_bytes(&xb).to_mont(curve),
        FujiField::from_bytes(&yb).to_mont(curve),
    )
}

#[cfg(test)]
mod tests {
    use crate::poly::commitment::{Params, MSM};
    use group::Curve;
    use pasta_curves::{arithmetic::CurveAffine, EpAffine, Fp, Fq};

    #[test]
    fn msm_arithmetic() {
        let base = EpAffine::from_xy(-Fp::one(), Fp::from(2)).unwrap();
        let base_viol = (base + base).to_affine();

        let params = Params::new(4);
        let mut a: MSM<EpAffine> = MSM::new(&params);
        a.append_term(Fq::one(), base);
        // a = [1] P
        assert!(!a.clone().eval());
        a.append_term(Fq::one(), base);
        // a = [1+1] P
        assert!(!a.clone().eval());
        a.append_term(-Fq::one(), base_viol);
        // a = [1+1] P + [-1] 2P
        assert!(a.clone().eval());
        let b = a.clone();

        // Append a point that is the negation of an existing one.
        a.append_term(Fq::from(4), -base);
        // a = [1+1-4] P + [-1] 2P
        assert!(!a.clone().eval());
        a.append_term(Fq::from(2), base_viol);
        // a = [1+1-4] P + [-1+2] 2P
        assert!(a.clone().eval());

        // Add two MSMs with common bases.
        a.scale(Fq::from(3));
        a.add_msm(&b);
        // a = [3*(1+1)+(1+1-4)] P + [3*(-1)+(-1+2)] 2P
        assert!(a.clone().eval());

        let mut c: MSM<EpAffine> = MSM::new(&params);
        c.append_term(Fq::from(2), base);
        c.append_term(Fq::one(), -base_viol);
        // c = [2] P + [1] (-2P)
        assert!(c.clone().eval());
        // Add two MSMs with bases that differ only in sign.
        a.add_msm(&c);
        assert!(a.eval());
    }
}
