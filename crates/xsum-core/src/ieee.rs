//! IEEE-754 binary64 layout constants and helpers.
//!
//! Throughout the crate a finite `f64` is viewed as the exact value
//! `±m · 2^(e − 1075)`, where `e` is the biased exponent and `m` the 53-bit
//! significand *including* the implicit leading bit. Subnormals use `e = 1` and
//! have no implicit bit. Every finite double is therefore an integer multiple
//! of `2^-1075`, which is the unit used by all superaccumulator digits.

/// Number of explicitly stored mantissa (fraction) bits.
pub(crate) const MANTISSA_BITS: u32 = 52;

/// Mask selecting the stored mantissa bits.
pub(crate) const MANTISSA_MASK: u64 = (1 << MANTISSA_BITS) - 1;

/// The implicit leading significand bit of normal numbers.
pub(crate) const IMPLICIT_BIT: u64 = 1 << MANTISSA_BITS;

/// Mask selecting the biased exponent after shifting out the mantissa.
pub(crate) const EXP_MASK: u64 = 0x7FF;

/// Biased exponent value used by infinities and NaNs.
pub(crate) const EXP_SPECIAL: u64 = EXP_MASK;

/// Largest biased exponent of a finite number.
pub(crate) const MAX_FINITE_EXP: u64 = EXP_SPECIAL - 1;

/// Sign bit of an `f64` bit pattern.
pub(crate) const SIGN_BIT: u64 = 1 << 63;

/// Returns the biased exponent field of an `f64` bit pattern.
#[inline]
pub(crate) const fn biased_exp(bits: u64) -> u64 {
    (bits >> MANTISSA_BITS) & EXP_MASK
}

/// Returns `true` if the bit pattern encodes an infinity or a NaN.
#[inline]
pub(crate) const fn is_special(bits: u64) -> bool {
    biased_exp(bits) == EXP_SPECIAL
}

/// A finite `f64` decomposed so that its value is
/// `(-1)^negative · significand · 2^(exponent − 1075)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Decomposed {
    /// Whether the sign bit is set.
    pub negative: bool,
    /// Biased exponent, at least 1 (subnormals are mapped to 1).
    pub exponent: u32,
    /// Significand including the implicit bit for normal numbers (< 2^53).
    pub significand: u64,
}

impl Decomposed {
    /// Decomposes the bit pattern of a *finite* `f64`.
    ///
    /// Zeros yield a zero significand. Must not be called for infinities or
    /// NaNs (checked in debug builds).
    #[inline]
    pub(crate) const fn from_finite_bits(bits: u64) -> Self {
        debug_assert!(!is_special(bits));
        let exp = biased_exp(bits);
        let mantissa = bits & MANTISSA_MASK;
        let (exponent, significand) = if exp == 0 {
            (1, mantissa)
        } else {
            (exp as u32, mantissa | IMPLICIT_BIT)
        };
        Self {
            negative: bits & SIGN_BIT != 0,
            exponent,
            significand,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decomposes_normal_numbers() {
        let d = Decomposed::from_finite_bits(1.0_f64.to_bits());
        assert_eq!(
            d,
            Decomposed {
                negative: false,
                exponent: 1023,
                significand: IMPLICIT_BIT
            }
        );
        // 1.0 == 2^52 · 2^(1023 − 1075)
        let d = Decomposed::from_finite_bits((-1.5_f64).to_bits());
        assert!(d.negative);
        assert_eq!(d.significand, IMPLICIT_BIT | (1 << 51));
    }

    #[test]
    fn decomposes_subnormals_and_zero() {
        let tiny = f64::from_bits(1);
        let d = Decomposed::from_finite_bits(tiny.to_bits());
        assert_eq!((d.exponent, d.significand), (1, 1));

        let d = Decomposed::from_finite_bits((-0.0_f64).to_bits());
        assert!(d.negative);
        assert_eq!(d.significand, 0);
    }

    #[test]
    fn classifies_specials() {
        assert!(is_special(f64::INFINITY.to_bits()));
        assert!(is_special(f64::NEG_INFINITY.to_bits()));
        assert!(is_special(f64::NAN.to_bits()));
        assert!(!is_special(f64::MAX.to_bits()));
        assert_eq!(biased_exp(f64::MAX.to_bits()), MAX_FINITE_EXP);
    }
}
