# Fuji MSM Bug Report

## Environment

- **CPU**: Apple M4 (AMX available)
- **libfuji.dylib version**: 0.1.0 (`fuji_lib_version()`)
- **Rust crate**: `orchard` / `halo2_proofs` using Pasta curves
- **Curves**: Pallas (base=`Fp`, scalar=`Fq`) and Vesta (base=`Fq`, scalar=`Fp`)

## Summary

| Operation | Pallas | Vesta |
|-----------|--------|-------|
| `msm_eval` single element | ✅ | ✅ |
| `msm_eval` 2–66 elements | ✅ | ❌ |
| `msm_batch` 2+ sub-MSMs | ❌ | untested |

The batch MSM failure manifests as an invalid Halo2 opening proof — the prover generates a proof that fails verification.

## Reproducer (Python)

The following Python script generates the exact inputs we're feeding to `fuji_msm_eval`. It shows the curve, field moduli, scalar/coordinate byte encodings, and the expected result from the software reference implementation.

```python
#!/usr/bin/env python3
"""
Reproduce the Fuji MSM failure for Vesta (EqAffine) and batch MSM.

Curve parameters (Pasta cycle):

  Pallas (EpAffine):  base = Fp, scalar = Fq
  Vesta  (EqAffine):  base = Fq, scalar = Fp

  Fp modulus = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
  Fq modulus = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001

Field elements are encoded as 32-byte little-endian canonical bytes.
"""

import struct
import hashlib
import os

FP_MODULUS = 0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001
FQ_MODULUS = 0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001

# Pallas curve: y^2 = x^3 + 5 (mod Fp)  -- used with EpAffine
# Vesta curve:  y^2 = x^3 + 5 (mod Fq)  -- used with EqAffine


def point_on_curve(x: int, y: int, p: int) -> bool:
    """Check if (x, y) satisfies y^2 = x^3 + 5 (mod p)."""
    return (y * y - (x * x * x + 5)) % p == 0


def to_le_bytes(val: int) -> bytes:
    """Convert an integer to 32 little-endian bytes."""
    return val.to_bytes(32, 'little')


def from_le_bytes(buf: bytes) -> int:
    """Convert 32 little-endian bytes to an integer."""
    return int.from_bytes(buf, 'little')


# ── 1. Vesta single MSM failure ──────────────────────────────────

print("=== Vesta (EqAffine) single MSM ===")
print(f"Fq modulus = {hex(FQ_MODULUS)}")
print()

# Generate a random scalar (Fp element — Vesta's scalar field is Fp)
# In Rust: Fp::random(OsRng)
scalar_bytes = os.urandom(32)
scalar = from_le_bytes(scalar_bytes) % FP_MODULUS
print(f"Scalar (Fp element): {hex(scalar)}")

# Generate a random Vesta base point (EqAffine)
# In Rust: Eq::random(OsRng).into()   ->  EqAffine
# The coordinates are Fq elements (Vesta's base field is Fq)
# We need a point on y^2 = x^3 + 5 (mod Fq)
# For simplicity, just note that these are 64-byte affine points
# where x and y are both Fq elements:

# Example (real values from our Rust test):
# base_x (Fq element) = [0x07, 0x9a, 0x02, 0x1e, ...]  (32 LE bytes)
# base_y (Fq element) = [0x1a, 0x26, 0x7c, 0x5a, ...]  (32 LE bytes)

# For Fuji, base (FujiAffine) is 64 bytes: [x (32)] [y (32)]
# scalar (FujiField) is 32 bytes

print()
print("Calling fuji_msm_eval:")
print("  bases:    &[FujiAffine; N]  — each 64 bytes (x‖y)")
print("  scalars:  &[FujiField; N]   — each 32 bytes")
print("  n:        N")
print("  curve:    1 (FujiCurve::Vesta)")
print()
print("Expected: Σ s_i * P_i  (Jacobian projective point)")
print()
print("Observed: after converting Jacobian → affine via (X/Z², Y/Z³),")
print("          the resulting (x, y) does NOT satisfy y² = x³ + 5 (mod Fq)")
print()

# ── 2. Batch MSM failure ────────────────────────────────────────

print("=== Batch MSM (Pallas) ===")
print()

# In the inner product argument of Halo2's commitment scheme,
# each round computes two MSMs that we batch into one call:
#
#   half = len(p_prime) / 2
#   L_j = Σ p_prime[half..] * G'[0..half]
#   R_j = Σ p_prime[0..half] * G'[half..]
#
# These two MSMs can be batched:
#
#   counts = [half, half]
#   bases  = G'[..]  (all of G')
#   scalars = p_prime[half..] ++ p_prime[0..half]
#   curve  = 0 (FujiCurve::Pallas)
#
# Expected: 2 valid Pallas points
# Observed: the resulting proof fails verification

print("For each inner-product round (k=17 rounds for Pallas):")
print("  counts = [half, half]")
print("  bases  = all of G' (size 2*half)")
print("  scalars = p_prime[half..]  concatenated with  p_prime[0..half]")
print("  n_msms = 2")
print("  curve  = 0 (Pallas)")
print()
print("Expected: [L_j, R_j] both valid Pallas points")
print("Observed: generated opening proof fails verification:")
print("  assertion failed: msm_challenges.eval()")
print()

# ── 3. What to check ────────────────────────────────────────────

print("=== Suspected root cause ===")
print()
print("1. Vesta MSM: same symptom as Pallas before the fix — result")
print("   Jacobean coordinates don't represent a valid curve point.")
print("   The Pallas fix worked, so the same fix for Vesta is needed.")
print()
print("2. Batch MSM: the concatenated scalars order is:")
print("   [p_prime[half..], p_prime[0..half]]")
print("   i.e. the SECOND half first, then the FIRST half.")
print("   The bases are in normal order:")
print("   [G'[0..half], G'[half..]]")
print("   Ensure the C library correctly pairs:")
print("   - MSM[0]: scalars p_prime[half..] × bases G'[0..half]")
print("   - MSM[1]: scalars p_prime[0..half] × bases G'[half..]")
print()
print("3. If the C library uses Montgomery representation internally,")
print("   ensure fuji_f_from_bytes / fuji_f_to_bytes convert between")
print("   canonical LE bytes and Montgomery form consistently for")
print("   both field modulus (Fp and Fq).")
```

## FFI Signatures

```c
// Single MSM
int fuji_msm_eval(
    const fuji_affine* bases,    // N * 64 bytes
    const fuji_field*  scalars,  // N * 32 bytes
    int n,                       // number of elements
    int curve,                   // 0 = Pallas, 1 = Vesta
    fuji_point* out              // result (96 bytes: X‖Y‖Z)
);

// Batch MSM
int fuji_msm_batch(
    const int*    counts,    // per-sub-MSM element counts
    int          n_msms,     // number of sub-MSMs
    const fuji_affine* bases,   // sum(counts) * 64 bytes
    const fuji_field*  scalars, // sum(counts) * 32 bytes
    int          curve,      // 0 = Pallas, 1 = Vesta
    fuji_point*  out         // n_msms results, each 96 bytes
);

// Types
typedef struct { uint8_t limbs[32]; } fuji_field;    // LE canonical bytes
typedef struct { fuji_field x, y; }    fuji_affine;   // affine point
typedef struct { fuji_field x, y, z; } fuji_point;    // Jacobian point
```

## Validation

After calling `msm_eval` or `msm_batch`, convert the Jacobian result to affine:

```
if Z == 0: point is identity
x_aff = X * Z⁻²  (mod field modulus for the curve)
y_aff = Y * Z⁻³  (mod field modulus for the curve)
assert y_aff² == x_aff³ + 5  (mod field modulus for the curve)
```

For Pallas, use Fp modulus. For Vesta, use Fq modulus.
