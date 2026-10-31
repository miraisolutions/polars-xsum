//! Rounding of an exact superaccumulator value to a floating-point number.
//!
//! The exact value is `N · 2^-1075` for an integer `N` held as base-`2^32`
//! digits. Rounding to `f64` means finding the 53 most significant bits of `|N|`
//! (but never resolving below `2^-1074`, the subnormal spacing), and deciding
//! the last bit from the guard bit and the sticky bit (the OR of everything
//! below the guard bit).

use crate::ieee::{IMPLICIT_BIT, MANTISSA_BITS, MANTISSA_MASK, MAX_FINITE_EXP, SIGN_BIT};
use crate::small::{DIGIT_BITS, NUM_DIGITS, propagate_carries};

/// Significand precision of `f64`, including the implicit bit.
const PRECISION: u32 = MANTISSA_BITS + 1;

/// How to resolve an inexact result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Rounding {
    /// IEEE round-to-nearest, ties to even.
    NearestEven,
    /// Round-to-odd: truncate, then set the last bit if anything was lost.
    ///
    /// Rounding the result once more to a format with at most `53 − 2` bits of
    /// precision (such as `f32`) gives the correctly rounded value. Results
    /// beyond the `f64` range become `±∞`.
    Odd,
}

/// Rounds the value represented by `digits` (in any carry state) to `f64`
/// with the rounding mode `mode`.
///
/// An exact zero yields `+0.0`.
pub(crate) fn round_digits(mut digits: [i64; NUM_DIGITS], mode: Rounding) -> f64 {
    propagate_carries(&mut digits);
    let negative = digits[NUM_DIGITS - 1] < 0;
    if negative {
        for digit in &mut digits {
            *digit = -*digit;
        }
        propagate_carries(&mut digits);
    }
    let magnitude = Magnitude::from_normalized(&digits).round(mode);
    f64::from_bits(if negative {
        magnitude | SIGN_BIT
    } else {
        magnitude
    })
}

/// A non-negative integer as little-endian base-`2^32` digits.
struct Magnitude {
    /// The digits, least significant first, each in `[0, 2^32)`.
    digits: [u32; NUM_DIGITS + 1],
}

impl Magnitude {
    /// Converts normalised, non-negative digits. The top digit may exceed
    /// `2^32` and is split into two.
    fn from_normalized(digits: &[i64; NUM_DIGITS]) -> Self {
        debug_assert!(digits.iter().all(|&d| d >= 0));
        let mut out = [0_u32; NUM_DIGITS + 1];
        for (o, &d) in out.iter_mut().zip(digits) {
            *o = d as u32;
        }
        out[NUM_DIGITS] = (digits[NUM_DIGITS - 1] >> DIGIT_BITS) as u32;
        Self { digits: out }
    }

    /// Number of significant bits (0 for zero).
    fn bit_len(&self) -> u32 {
        self.digits.iter().rposition(|&d| d != 0).map_or(0, |k| {
            k as u32 * DIGIT_BITS + (DIGIT_BITS - self.digits[k].leading_zeros())
        })
    }

    /// Returns `count ≤ 64` bits starting at bit position `low`.
    fn bits(&self, low: u32, count: u32) -> u64 {
        debug_assert!(count <= 64);
        let first = (low / DIGIT_BITS) as usize;
        let window = self
            .digits
            .iter()
            .skip(first)
            .take(3)
            .enumerate()
            .fold(0_u128, |acc, (k, &d)| {
                acc | u128::from(d) << (k as u32 * DIGIT_BITS)
            });
        let shifted = window >> (low % DIGIT_BITS);
        (shifted & ((1_u128 << count) - 1)) as u64
    }

    /// Returns `true` if any bit below position `position` is set.
    fn any_below(&self, position: u32) -> bool {
        let full = (position / DIGIT_BITS) as usize;
        let partial_mask = (1_u32 << (position % DIGIT_BITS)) - 1;
        self.digits[..full].iter().any(|&d| d != 0) || self.digits[full] & partial_mask != 0
    }

    /// Rounds `self · 2^-1075` to an `f64` with the rounding mode `mode` and
    /// returns its (unsigned) bit pattern.
    fn round(&self, mode: Rounding) -> u64 {
        let len = self.bit_len();
        if len == 0 {
            return 0;
        }
        // The result is `q · 2^(shift − 1075)`. `shift ≥ 1` because the
        // smallest representable spacing is 2^-1074, i.e. two units.
        let mut shift = len.saturating_sub(PRECISION).max(1);
        let mut q = self.bits(shift, PRECISION);
        let guard = self.bits(shift - 1, 1) == 1;
        let sticky = self.any_below(shift - 1);

        match mode {
            Rounding::NearestEven => {
                if guard && (sticky || q & 1 == 1) {
                    q += 1;
                    if q == IMPLICIT_BIT << 1 {
                        q >>= 1;
                        shift += 1;
                    }
                }
            }
            Rounding::Odd => {
                if guard || sticky {
                    q |= 1;
                }
            }
        }

        if u64::from(shift) > MAX_FINITE_EXP {
            f64::INFINITY.to_bits()
        } else if q >= IMPLICIT_BIT {
            // Normal number: `q · 2^(shift − 1075)` has biased exponent `shift`.
            (u64::from(shift) << MANTISSA_BITS) | (q & MANTISSA_MASK)
        } else {
            // Subnormal (only possible with `shift == 1`): `q · 2^-1074`.
            q
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds digits for `n · 2^-1075` with `n = value · 2^shift`.
    fn digits_of(value: u128, shift: u32) -> [i64; NUM_DIGITS] {
        let mut digits = [0; NUM_DIGITS];
        let first = (shift / DIGIT_BITS) as usize;
        let wide = value << (shift % DIGIT_BITS);
        for k in 0..4 {
            if first + k < NUM_DIGITS {
                digits[first + k] = ((wide >> (32 * k)) & 0xFFFF_FFFF) as i64;
            }
        }
        digits
    }

    fn round(value: u128, shift: u32) -> f64 {
        round_digits(digits_of(value, shift), Rounding::NearestEven)
    }

    #[test]
    fn zero_is_positive() {
        let r = round_digits([0; NUM_DIGITS], Rounding::NearestEven);
        assert_eq!(r.to_bits(), 0);
    }

    #[test]
    fn exact_values() {
        // 1.0 = 2^1075 units.
        assert_eq!(round(1, 1075), 1.0);
        assert_eq!(round(3, 1074), 1.5);
        // Smallest subnormal = 2 units, largest subnormal and smallest normal.
        assert_eq!(round(2, 0), f64::from_bits(1));
        assert_eq!(round((1 << 52) - 1, 1), f64::from_bits((1 << 52) - 1));
        assert_eq!(round(1, 53), f64::MIN_POSITIVE);
        assert_eq!(round((1 << 53) - 1, 2046), f64::MAX);
    }

    #[test]
    fn ties_go_to_even() {
        // 1 + 2^-53 is half-way between 1 and the next double: round down.
        assert_eq!(round((1 << 53) + 1, 1075 - 53), 1.0);
        // 1 + 3·2^-53 is half-way between 1+2^-52 and 1+2^-51: round up.
        assert_eq!(round((1 << 53) + 3, 1075 - 53), 1.0 + 2f64.powi(-51));
        // Slightly above the tie: round up.
        assert_eq!(round((1 << 54) + 3, 1075 - 54), 1.0 + 2f64.powi(-52));
        // Half of the smallest subnormal (one unit) rounds to zero; 3 units to 4.
        assert_eq!(round(1, 0), 0.0);
        assert_eq!(round(3, 0), f64::from_bits(2));
    }

    #[test]
    fn rounding_carries_into_next_binade_and_overflow() {
        // (2^53 − 1/2)·2^k rounds up to 2^(53+k).
        assert_eq!(round((1 << 54) - 1, 1075 - 1), 2f64.powi(53));
        // MAX + half an ulp is a tie with an odd significand: overflow.
        assert_eq!(round((1 << 54) - 1, 970 + 1075), f64::INFINITY);
        // Just below the tie stays at MAX.
        assert_eq!(round((1 << 55) - 3, 969 + 1075), f64::MAX);
    }

    #[test]
    fn negative_values() {
        let mut digits = digits_of(3, 1074);
        for d in &mut digits {
            *d = -*d;
        }
        assert_eq!(round_digits(digits, Rounding::NearestEven), -1.5);
    }

    #[test]
    fn round_to_odd_sets_sticky_bit() {
        let exact = digits_of((1 << 53) + 1, 1075 - 53);
        assert_eq!(round_digits(exact, Rounding::Odd), 1.0 + f64::EPSILON);
        let exact = digits_of(1 << 53, 1075 - 53);
        assert_eq!(round_digits(exact, Rounding::Odd), 1.0);
    }
}
