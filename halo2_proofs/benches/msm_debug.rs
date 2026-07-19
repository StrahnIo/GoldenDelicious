use ff::PrimeField;
use halo2_proofs::arithmetic::CurveAffine;
use pasta_curves::{Ep, EpAffine, Fp, Fq};

fn hex_to_bytes(hex: &str) -> [u8; 32] {
    let hex = hex.replace(' ', "");
    let mut bytes = [0u8; 32];
    for i in 0..32 {
        bytes[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    bytes
}

fn main() {
    // Generator G coordinates (from kernel dump, LE bytes)
    // X = 00 00 00 00 ed 30 2d 99 1b f9 4c 09 fc 98 46 22
    //     00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40
    // Y = 02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    //     00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00
    let g_x = Fp::from_repr(hex_to_bytes(
        "00 00 00 00 ed 30 2d 99 1b f9 4c 09 fc 98 46 22 \
         00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 40",
    ))
    .unwrap();
    let g_y = Fp::from_repr(hex_to_bytes(
        "02 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 \
         00 00 00 00 00 00 00 00 00 00 00 00 00 00 00 00",
    ))
    .unwrap();
    let gen = EpAffine::from_xy(g_x, g_y).unwrap();
    eprintln!("Generator: {:?}", gen.coordinates().unwrap());

    // Scalar s (from kernel dump, LE bytes)
    // s = 83 2f f0 92 7d 8a da ef 3a e3 a5 12 ff 90 e9 76
    //     02 4f af b4 34 70 b5 7b 4e b0 61 b7 5f a7 62 1c
    let s = Fq::from_repr(hex_to_bytes(
        "83 2f f0 92 7d 8a da ef 3a e3 a5 12 ff 90 e9 76 \
         02 4f af b4 34 70 b5 7b 4e b0 61 b7 5f a7 62 1c",
    ))
    .unwrap();

    // Compute s·G using best_multiexp (same as SW bench)
    use halo2_proofs::arithmetic::best_multiexp;
    let result = best_multiexp::<EpAffine>(&[s], &[gen]);

    // Convert to affine and print coordinates
    use group::Curve;
    let result_affine = result.to_affine();
    let coords = result_affine.coordinates().unwrap();
    eprintln!("\nResult = s·G (affine, normal form):");
    eprintln!("X = {:02x?}", coords.x().to_repr().as_ref());
    eprintln!("Y = {:02x?}", coords.y().to_repr().as_ref());

    // Compare with kernel output
    let kernel_x = hex_to_bytes(
        "4f e3 ce 74 4c 63 fb a7 45 c2 32 7d 5c e4 c8 46 \
         72 87 ed 54 36 77 2d 73 ad ca 65 6a a9 e5 c1 2b",
    );
    let kernel_y = hex_to_bytes(
        "d6 dc b2 0b 39 bf 49 33 28 24 e9 09 5c 32 fb 8c \
         b2 ce 36 15 8b 9c 43 12 00 f0 97 65 fa 52 96 2c",
    );

    let x_match = coords.x().to_repr().as_ref() == kernel_x;
    let y_match = coords.y().to_repr().as_ref() == kernel_y;
    if x_match && y_match {
        eprintln!("\n✅ MATCH — pasta_curves result matches kernel!");
    } else {
        eprintln!("\n❌ MISMATCH — pasta_curves result differs from kernel:");
        eprintln!("  Kernel X: {:02x?}", &kernel_x[..]);
        eprintln!("  Got    X: {:02x?}", coords.x().to_repr().as_ref());
        eprintln!("  Kernel Y: {:02x?}", &kernel_y[..]);
        eprintln!("  Got    Y: {:02x?}", coords.y().to_repr().as_ref());
    }

    // Also print as hex string for easy copy-paste
    eprintln!("\nHex string (for kernel comparison):");
    eprint!("result_x = \"");
    for b in coords.x().to_repr().as_ref() {
        eprint!("{:02x} ", b);
    }
    eprintln!("\"");
    eprint!("result_y = \"");
    for b in coords.y().to_repr().as_ref() {
        eprint!("{:02x} ", b);
    }
    eprintln!("\"");
}
