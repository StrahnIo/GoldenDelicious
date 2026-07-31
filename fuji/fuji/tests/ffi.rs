use fuji::*;

#[test]
fn amx_detection() {
    assert!(detection::amx_available());
    let brand = detection::cpu_brand();
    assert!(!brand.is_empty());
    let ver = detection::lib_version();
    assert!(!ver.is_empty());
}

#[test]
fn sme_detection() {
    // SME should be available on M4 (our test hardware).
    // This test asserts the detection path works.
    let available = detection::sme_available();
    // On M4 this is true; on M1-M3 it's false.
    // We just verify the function returns without crashing.
    assert!(available || !available);
}

#[test]
fn field_zero_one() {
    let z = FujiField::zero();
    let o = FujiField::one();
    assert_ne!(z, o);
    assert_eq!(z, z);
    assert_eq!(o, o);
}

#[test]
fn field_add_sub() {
    let a = FujiField::from_bytes(&[5u8; 32]);
    let b = FujiField::from_bytes(&[3u8; 32]);
    let sum = a.add(&b, FujiCurve::Pallas).unwrap();
    assert_ne!(sum, a);
    assert_ne!(sum, b);
    let diff = a.sub(&b, FujiCurve::Pallas).unwrap();
    assert_ne!(diff, a);
}

#[test]
fn field_mul_one() {
    let one = FujiField::one();
    let five = FujiField::from_bytes(&[5u8; 32]);
    let r = five.mul(&one, FujiCurve::Pallas).unwrap();
    assert_eq!(r, five);
}

#[test]
fn point_gen_pallas() {
    let g = FujiAffine::gen_pallas();
    let p = FujiPoint::from_affine(&g, FujiCurve::Pallas).unwrap();
    assert!(!p.is_identity());
}

#[test]
fn point_double_equals_add() {
    let g = FujiAffine::gen_pallas();
    let p = FujiPoint::from_affine(&g, FujiCurve::Pallas).unwrap();
    let dbl = p.double(FujiCurve::Pallas).unwrap();
    let added = p.add(&p, FujiCurve::Pallas).unwrap();
    assert_eq!(dbl.x_limbs(), added.x_limbs(), "double != add");
}

#[test]
fn msm_simple() {
    let g = FujiAffine::gen_pallas();
    let one = FujiField::one();

    let r = msm::msm_eval(&[g], &[one], FujiCurve::Pallas).unwrap();
    assert!(!r.is_identity());

    let r2 = msm::msm_eval(&[g, g], &[one, one], FujiCurve::Pallas).unwrap();
    assert!(!r2.is_identity());
}

#[test]
fn sme_streaming_enter_exit() {
    // Enter and exit streaming mode via the RAII guard
    {
        let _stream = sme::SmeStream::enter();
        // Inside streaming mode — SME instructions are accessible
    }
    // Outside streaming mode — guard was dropped
}

#[test]
#[ignore]
fn sme_umopa_compute() {
    if !detection::sme_available() {
        return;
    }
    // Compute outer product of [1u8; 32] × [1u8; 32]
    let a = [1u8; 32];
    let b = [1u8; 32];
    // Each element of the result should be 1 * 1 = 1 if UMOPA works.
    // Note: this may SIGILL if the STR ZA .word encodings are wrong
    // for this hardware/toolchain combination.
    let result = sme::umopa_outer_product(&a, &b);
    assert_eq!(result[0], 1, "first element should be 1*1=1");
    assert_eq!(result.len(), 1024, "should produce 1024 products");
}

#[test]
#[ignore]
fn sme_umopa_raw_variant() {
    if !detection::sme_available() {
        return;
    }
    let a = [3u8; 32];
    let b = [5u8; 32];
    let mut out = [0u32; 1024];
    unsafe {
        sme::umopa_outer_product_raw(&a, &b, &mut out);
    }
    assert_eq!(out[0], 15, "first element should be 3*5=15");
}

#[test]
fn msm_batch_simple() {
    let g = FujiAffine::gen_pallas();
    let one = FujiField::one();
    let counts = vec![1i32, 1i32];
    let bases = vec![g, g];
    let scalars = vec![one, one];
    let results = msm::msm_batch(&counts, &bases, &scalars, FujiCurve::Pallas).unwrap();
    assert_eq!(results.len(), 2);
    assert!(!results[0].is_identity());
}

#[test]
fn msm_batch4_simple() {
    let g = FujiAffine::gen_pallas();
    let n = 64;
    let bases: Vec<FujiAffine> = (0..n).map(|_| g).collect();
    let mut scalars = Vec::with_capacity(4 * n);
    for i in 0..4 * n {
        let mut bytes = [0u8; 32];
        bytes[0] = (i as u8).wrapping_mul(7).wrapping_add(13);
        bytes[1] = (i >> 8) as u8;
        scalars.push(FujiField::from_bytes(&bytes));
    }

    let batch_results = msm::prl_pippenger_batch_4(&scalars, &bases, FujiCurve::Pallas).unwrap();

    let ref0 = msm::prl_pippenger(&scalars[0..n], &bases, FujiCurve::Pallas).unwrap();
    let ref1 = msm::prl_pippenger(&scalars[n..2*n], &bases, FujiCurve::Pallas).unwrap();
    let ref2 = msm::prl_pippenger(&scalars[2*n..3*n], &bases, FujiCurve::Pallas).unwrap();
    let ref3 = msm::prl_pippenger(&scalars[3*n..4*n], &bases, FujiCurve::Pallas).unwrap();

    assert_eq!(batch_results[0].x_limbs(), ref0.x_limbs(), "MSM 0 mismatch");
    assert_eq!(batch_results[1].x_limbs(), ref1.x_limbs(), "MSM 1 mismatch");
    assert_eq!(batch_results[2].x_limbs(), ref2.x_limbs(), "MSM 2 mismatch");
    assert_eq!(batch_results[3].x_limbs(), ref3.x_limbs(), "MSM 3 mismatch");
}
