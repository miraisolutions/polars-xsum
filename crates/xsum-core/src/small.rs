//! The *small superaccumulator* (Neal 2015, section 3).
//!
//! The exact sum is held as a vector of signed 64-bit "digits" in base `2^32`:
//! digit `k` has weight `2^(32·k − 1075)`. A summand with biased exponent `e`
//! touches only digits `e / 32` and `e / 32 + 1`, and its significand is
//! shifted by `e mod 32`. Each digit has 31 spare bits above its 32 "home" bits,
//! so many summands can be added before carries must be propagated. Carry
//! propagation normalises every digit except the topmost one into `[0, 2^32)`.

use crate::ieee::{Decomposed, EXP_SPECIAL, MANTISSA_BITS, MANTISSA_MASK, is_special};
use crate::round::{Rounding, round_digits};

/// Bits per digit, i.e. the base of the digit representation is `2^DIGIT_BITS`.
pub(crate) const DIGIT_BITS: u32 = 32;

/// `log2(DIGIT_BITS)`: the low exponent bits that select the shift within a digit.
const LOW_EXP_BITS: u32 = DIGIT_BITS.trailing_zeros();

/// Mask selecting the home bits of a digit.
const DIGIT_MASK: i64 = (1 << DIGIT_BITS) - 1;

/// Number of digits.
///
/// Finite exponents select digits `0..=63`. A summand spans up to four
/// consecutive digits: two for a single double, and up to four for the
/// combined chunks transferred from the large accumulator.
pub(crate) const NUM_DIGITS: usize = (EXP_SPECIAL as usize >> LOW_EXP_BITS) + 4;

/// Number of additions that may be performed between two carry propagations.
///
/// After propagation a digit is in `[0, 2^32)`. Every addition changes a digit
/// by less than `2^53` in magnitude (see [`SmallAccumulator::add_finite_bits`]),
/// so the digit must stay within `i64` for that many additions.
const ADDS_BETWEEN_PROPAGATIONS: u32 = (1 << (63 - (MANTISSA_BITS + 1))) - 1;

const _: () = {
    let max_change = ADDS_BETWEEN_PROPAGATIONS as i128 * (1_i128 << (MANTISSA_BITS + 1));
    assert!(DIGIT_MASK as i128 + max_change <= i64::MAX as i128);
    assert!(-max_change >= i64::MIN as i128);
};

/// Accumulated infinities and NaNs, which are kept apart from the finite sum.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Special {
    /// No infinity or NaN has been added.
    #[default]
    None,
    /// Only infinities of a single sign (given as bit pattern) have been added.
    Infinity(u64),
    /// The result is NaN: a NaN was added (its bit pattern is kept), or
    /// infinities of both signs were added.
    NaN(u64),
}

impl Special {
    /// Records the bit pattern of an infinity or NaN.
    #[must_use]
    pub(crate) fn absorb(self, bits: u64) -> Self {
        debug_assert!(is_special(bits));
        let is_nan = bits & MANTISSA_MASK != 0;
        match self {
            Self::NaN(_) => self,
            _ if is_nan => Self::NaN(bits),
            Self::None => Self::Infinity(bits),
            Self::Infinity(inf) if inf == bits => self,
            Self::Infinity(_) => Self::NaN(f64::NAN.to_bits()),
        }
    }

    /// Combines the specials of two partial sums.
    #[must_use]
    pub(crate) fn combine(self, other: Self) -> Self {
        match other {
            Self::None => self,
            Self::Infinity(bits) | Self::NaN(bits) => self.absorb(bits),
        }
    }

    /// The overall result if it is an infinity or NaN.
    pub(crate) fn value(self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::Infinity(bits) | Self::NaN(bits) => Some(f64::from_bits(bits)),
        }
    }
}

/// Normalises `digits` so that all but the top digit lie in `[0, 2^32)`.
///
/// The represented value is unchanged; the sign of the whole number is the
/// sign of the (unnormalised) top digit afterwards.
pub(crate) fn propagate_carries(digits: &mut [i64; NUM_DIGITS]) {
    let (top, lower) = digits.split_last_mut().expect("NUM_DIGITS is non-zero");
    let mut carry = 0_i64;
    for digit in lower {
        let value = *digit + carry;
        *digit = value & DIGIT_MASK;
        carry = value >> DIGIT_BITS;
    }
    *top += carry;
}

/// An exact accumulator for sums of `f64` values with a compact (≈ 0.5 KiB)
/// state.
///
/// Adding a value costs a handful of integer instructions, and occasionally
/// (once every 1023 additions) a carry propagation over all 67 digits. The
/// accumulator never loses information, so the final result is the exact sum
/// rounded once.
///
/// The small accumulator is the fastest choice for short inputs. For long
/// inputs, [`LargeAccumulator`](crate::LargeAccumulator) is faster.
///
/// # Examples
///
/// ```
/// use xsum_core::SmallAccumulator;
///
/// let mut acc = SmallAccumulator::new();
/// acc.add(1e100);
/// acc.add(1.0);
/// acc.add(-1e100);
/// assert_eq!(acc.to_f64(), 1.0);
///
/// let acc: SmallAccumulator = [0.1; 10].into_iter().collect();
/// assert_eq!(acc.to_f64(), 1.0);
/// ```
#[derive(Clone, Debug)]
pub struct SmallAccumulator {
    /// Base-`2^32` digits; digit `k` has weight `2^(32·k − 1075)`.
    digits: [i64; NUM_DIGITS],
    /// Additions left before carries must be propagated.
    adds_until_propagate: u32,
    /// Infinities and NaNs seen so far.
    special: Special,
}

impl Default for SmallAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl SmallAccumulator {
    /// Creates an accumulator representing an empty sum (zero).
    #[must_use]
    pub const fn new() -> Self {
        Self {
            digits: [0; NUM_DIGITS],
            adds_until_propagate: ADDS_BETWEEN_PROPAGATIONS,
            special: Special::None,
        }
    }

    /// Adds a single value.
    #[inline]
    pub fn add(&mut self, value: f64) {
        if self.adds_until_propagate == 0 {
            self.propagate();
        }
        self.add_bits_unchecked(value.to_bits());
        self.adds_until_propagate -= 1;
    }

    /// Adds all values of a slice.
    ///
    /// Accepts any element type that converts losslessly into `f64`, in
    /// particular `f32` and `f64`.
    pub fn add_slice<T: Copy + Into<f64>>(&mut self, values: &[T]) {
        let mut rest = values;
        while !rest.is_empty() {
            if self.adds_until_propagate == 0 {
                self.propagate();
            }
            let n = rest.len().min(self.adds_until_propagate as usize);
            let (batch, tail) = rest.split_at(n);
            for &value in batch {
                self.add_bits_unchecked(value.into().to_bits());
            }
            self.adds_until_propagate -= n as u32;
            rest = tail;
        }
    }

    /// Adds another accumulator's sum to this one.
    pub fn merge(&mut self, other: &Self) {
        let mut other_digits = other.digits;
        propagate_carries(&mut other_digits);
        self.propagate();
        for (digit, other) in self.digits.iter_mut().zip(other_digits) {
            *digit += other;
        }
        self.propagate();
        self.special = self.special.combine(other.special);
    }

    /// Returns the exact sum correctly rounded to the nearest `f64`, with ties
    /// going to even.
    ///
    /// * Any NaN summand, or infinities of opposite signs, give NaN.
    /// * Otherwise an infinite summand gives that infinity.
    /// * An exact sum outside the finite `f64` range gives `±∞`.
    /// * An exact zero gives `+0.0`.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.special
            .value()
            .unwrap_or_else(|| round_digits(self.digits, Rounding::NearestEven))
    }

    /// Returns the exact sum correctly rounded to the nearest `f32`, with ties
    /// going to even.
    ///
    /// The exact sum is first rounded to `f64` with *round-to-odd*, which makes
    /// the second rounding to `f32` correct (no double-rounding error). Special
    /// values follow the same rules as [`to_f64`](Self::to_f64).
    #[must_use]
    pub fn to_f32(&self) -> f32 {
        let wide = self
            .special
            .value()
            .unwrap_or_else(|| round_digits(self.digits, Rounding::Odd));
        wide as f32
    }

    /// Adds `±magnitude · 2^(exponent − 1075)` for a magnitude below `2^96`.
    ///
    /// Used by the large accumulator to transfer pre-summed chunks.
    pub(crate) fn add_scaled(&mut self, magnitude: u128, exponent: u32, negative: bool) {
        debug_assert!(magnitude < 1 << 96);
        if self.adds_until_propagate == 0 {
            self.propagate();
        }
        self.adds_until_propagate -= 1;

        let index = (exponent >> LOW_EXP_BITS) as usize;
        let shifted = magnitude << (exponent & (DIGIT_BITS - 1));
        let pieces: [i64; 4] =
            std::array::from_fn(|k| ((shifted >> (k as u32 * DIGIT_BITS)) as i64) & DIGIT_MASK);
        for (digit, piece) in self.digits[index..index + 4].iter_mut().zip(pieces) {
            if negative {
                *digit -= piece;
            } else {
                *digit += piece;
            }
        }
    }

    /// Records an infinity or NaN.
    pub(crate) fn add_special(&mut self, bits: u64) {
        self.special = self.special.absorb(bits);
    }

    /// Adds a value without checking the carry budget.
    #[inline]
    fn add_bits_unchecked(&mut self, bits: u64) {
        if is_special(bits) {
            self.add_special(bits);
        } else {
            self.add_finite_bits(bits);
        }
    }

    /// Adds a finite value given by its bit pattern.
    ///
    /// The significand `m < 2^53` is shifted left by `s = e mod 32`, giving a
    /// number below `2^85` that is split into a low part (`< 2^32`) for digit
    /// `e / 32` and a high part (`< 2^53`) for the digit above.
    #[inline]
    fn add_finite_bits(&mut self, bits: u64) {
        let Decomposed {
            negative,
            exponent,
            significand,
        } = Decomposed::from_finite_bits(bits);
        let index = (exponent >> LOW_EXP_BITS) as usize;
        let shift = exponent & (DIGIT_BITS - 1);
        let low = ((significand << shift) as i64) & DIGIT_MASK;
        let high = (significand >> (DIGIT_BITS - shift)) as i64;
        let (low, high) = if negative { (-low, -high) } else { (low, high) };
        self.digits[index] += low;
        self.digits[index + 1] += high;
    }

    fn propagate(&mut self) {
        propagate_carries(&mut self.digits);
        self.adds_until_propagate = ADDS_BETWEEN_PROPAGATIONS;
    }
}

impl<T: Into<f64>> Extend<T> for SmallAccumulator {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        for value in iter {
            self.add(value.into());
        }
    }
}

impl<T: Into<f64>> FromIterator<T> for SmallAccumulator {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut acc = Self::new();
        acc.extend(iter);
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(values: &[f64]) -> f64 {
        let mut acc = SmallAccumulator::new();
        acc.add_slice(values);
        acc.to_f64()
    }

    #[test]
    fn digit_layout_constants() {
        assert_eq!(NUM_DIGITS, 67);
        assert_eq!(LOW_EXP_BITS, 5);
        assert_eq!(ADDS_BETWEEN_PROPAGATIONS, 1023);
    }

    #[test]
    fn propagation_normalises_lower_digits() {
        let mut acc = SmallAccumulator::new();
        for i in 0..5000 {
            acc.add(if i % 3 == 0 { -1.5e300 } else { 0.75e300 });
            acc.add(f64::from(i) * -1e-300);
        }
        acc.propagate();
        let (top, lower) = acc.digits.split_last().unwrap();
        assert!(lower.iter().all(|&d| (0..=DIGIT_MASK).contains(&d)));
        assert_eq!(*top, -1, "negative sums have an all-ones top digit");
    }

    #[test]
    fn carry_budget_boundary() {
        // Exactly the budget, budget + 1 and many multiples of it.
        let budget = ADDS_BETWEEN_PROPAGATIONS as usize;
        for n in [budget - 1, budget, budget + 1, 2 * budget + 1, 10 * budget] {
            // A single IEEE multiplication is the correctly rounded exact product.
            let value = 1.234_567_890_123_456_7e300;
            let values = vec![value; n];
            assert_eq!(sum(&values), value * n as f64, "n = {n}");
            let mut acc = SmallAccumulator::new();
            for &v in &values {
                acc.add(-v);
            }
            assert_eq!(acc.to_f64(), -value * n as f64, "n = {n}");
        }
    }

    #[test]
    fn add_and_add_slice_agree() {
        let values: Vec<f64> = (0..3000)
            .map(|i| f64::from(i).sin() * 2f64.powi(i % 200 - 100))
            .collect();
        let one_by_one: SmallAccumulator = values.iter().copied().collect();
        assert_eq!(one_by_one.to_f64().to_bits(), sum(&values).to_bits());
    }

    #[test]
    fn merge_matches_sequential_sum() {
        let values: Vec<f64> = (1..=4000).map(|i| 1.0 / f64::from(i)).collect();
        let (left, right) = values.split_at(1234);
        let mut a: SmallAccumulator = left.iter().copied().collect();
        let b: SmallAccumulator = right.iter().map(|v| -v).collect();
        a.merge(&b);
        let expected: SmallAccumulator = left
            .iter()
            .copied()
            .chain(right.iter().map(|v| -v))
            .collect();
        assert_eq!(a.to_f64().to_bits(), expected.to_f64().to_bits());
    }

    #[test]
    fn scaled_add_matches_repeated_add() {
        // 3 · 2^60 · 2^(1500 − 1075) added in one step.
        let mut scaled = SmallAccumulator::new();
        scaled.add_scaled(3 << 60, 1500, true);
        let unit = f64::from_bits(1500_u64 << MANTISSA_BITS) / 2f64.powi(52);
        assert_eq!(scaled.to_f64(), -3.0 * 2f64.powi(60) * unit);
        // Magnitudes beyond 64 bits, at the highest exponent.
        let mut wide = SmallAccumulator::new();
        wide.add_scaled(1 << 95, 1, false);
        assert_eq!(wide.to_f64(), 2f64.powi(95 + 1 - 1075));
        let mut top = SmallAccumulator::new();
        top.add_scaled(1 << 90, 2046, false);
        top.add_scaled(1 << 90, 2046, true);
        top.add_scaled(1 << 52, 2046, false);
        assert_eq!(top.to_f64(), 2f64.powi(1023));
    }

    #[test]
    fn special_values() {
        assert!(sum(&[1.0, f64::NAN]).is_nan());
        assert!(sum(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
        assert!(sum(&[f64::NAN, f64::INFINITY]).is_nan());
        assert_eq!(sum(&[f64::INFINITY, 1.0, f64::INFINITY]), f64::INFINITY);
        assert_eq!(sum(&[f64::NEG_INFINITY, f64::MAX]), f64::NEG_INFINITY);
    }

    #[test]
    fn special_combination_rules() {
        let pos = f64::INFINITY.to_bits();
        let neg = f64::NEG_INFINITY.to_bits();
        let nan = f64::NAN.to_bits() | 0x1234;
        assert_eq!(Special::None.absorb(pos), Special::Infinity(pos));
        assert_eq!(Special::Infinity(pos).absorb(pos), Special::Infinity(pos));
        assert!(matches!(
            Special::Infinity(pos).absorb(neg),
            Special::NaN(_)
        ));
        assert_eq!(Special::Infinity(pos).absorb(nan), Special::NaN(nan));
        assert_eq!(Special::NaN(nan).absorb(pos), Special::NaN(nan));
        assert_eq!(
            Special::None.combine(Special::Infinity(neg)),
            Special::Infinity(neg)
        );
        assert!(matches!(
            Special::Infinity(pos).combine(Special::Infinity(neg)),
            Special::NaN(_)
        ));
    }
}
