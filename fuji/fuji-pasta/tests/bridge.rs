use fuji_pasta::*;

#[test]
fn field_roundtrip_pallas() {
    let a = FujiField::<PallasTag>::from(42u64);
    let b: pasta_curves::Fp = a.into();
    let c: FujiField<PallasTag> = b.into();
    assert_eq!(a, c);
}

#[test]
fn field_roundtrip_vesta() {
    let a = FujiField::<VestaTag>::from(99u64);
    let b: pasta_curves::Fq = a.into();
    let c: FujiField<VestaTag> = b.into();
    assert_eq!(a, c);
}

#[test]
fn field_add_pallas() {
    let one = FujiField::<PallasTag>::one();
    let two = one + one;
    let three = two + one;
    assert_eq!(three, FujiField::<PallasTag>::from(3u64));
}

#[test]
fn field_mul_pallas() {
    let a = FujiField::<PallasTag>::from(5u64);
    let b = FujiField::<PallasTag>::from(7u64);
    assert_eq!(a * b, FujiField::<PallasTag>::from(35u64));
}

#[test]
fn group_generator_pallas() {
    use group::Group;
    let g = <FujiPoint<PallasTag> as Group>::generator();
    assert!(!bool::from(g.is_identity()));
    let dbl = g.double();
    assert!(!bool::from(dbl.is_identity()));
}

#[test]
fn group_scalar_mul_pallas() {
    use group::Group;
    let g = <FujiPoint<PallasTag> as Group>::generator();
    let one = FujiField::<VestaTag>::from(1u64);
    let result = g * one;
    assert!(!bool::from(result.is_identity()));
}

#[test]
fn ff_field_basics() {
    let a = FujiField::<PallasTag>::from(10u64);
    assert_eq!(a.double(), a + a);
    assert_eq!(a.square(), a * a);
    assert_eq!(a * FujiField::<PallasTag>::one(), a);
}
