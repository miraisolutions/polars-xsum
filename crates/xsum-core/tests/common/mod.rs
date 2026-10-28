//! Shared helpers for the integration tests: an exact reference implementation
//! based on arbitrary-precision integers and all summation variants under test.

#![allow(dead_code, reason = "each test binary uses a different subset")]

use num_bigint::BigInt;
use num_traits::{Signed, Zero};
use xsum_core::{LargeAccumulator, SmallAccumulator};

/// Exponent of the unit in which exact values are expressed: every finite
/// `f64` is an integer multiple of `2^-UNIT_EXP`.
const UNIT_EXP: u32 = 1075;

/// The exact value of a finite `f64` in units of `2^-1075`.
pub fn exact(x: f64) -> BigInt {
    assert!(x.is_finite(), "exact() requires a finite value, got {x}");
    let bits = x.to_bits();
    let exp = (bits >> 52) & 0x7FF;
    let mantissa = bits & ((1 << 52) - 1);
    let (exp, significand) = if exp == 0 {
        (1, mantissa)
    } else {
        (exp, mantissa | 1 << 52)
    };
    let magnitude = BigInt::from(significand) << exp;
    if x.is_sign_negative() {
        -magnitude
    } else {
        magnitude
    }
}

/// The exact sum of finite values in units of `2^-1075`.
pub fn exact_sum(values: &[f64]) -> BigInt {
    values.iter().map(|&x| exact(x)).sum()
}

/// The value a candidate result stands for; infinities stand for the first
/// power of two beyond the format's range, as in IEEE-754 overflow rounding.
fn candidate_value(r: f64, overflow_exp: u32) -> BigInt {
    if r.is_infinite() {
        let magnitude = BigInt::from(1) << (overflow_exp + UNIT_EXP);
        if r < 0.0 { -magnitude } else { magnitude }
    } else {
        exact(r)
    }
}

/// Checks that `r` is the exact value `n` rounded to nearest, ties to even,
/// by comparing distances to the neighbouring floats. `neighbours` returns the
/// adjacent representable values, `is_even` tests the last significand bit.
fn is_nearest_even(
    n: &BigInt,
    r: f64,
    overflow_exp: u32,
    neighbours: [f64; 2],
    is_even: bool,
) -> bool {
    if r.is_nan() {
        return false;
    }
    if n.is_zero() {
        return r.to_bits() == 0; // exact zero must be +0.0
    }
    let distance = |c: f64| (n - candidate_value(c, overflow_exp)).abs();
    let own = distance(r);
    neighbours
        .into_iter()
        .filter(|c| c.to_bits() != r.to_bits())
        .all(|c| {
            let other = distance(c);
            own < other || (own == other && is_even)
        })
}

/// Is `r` the exact sum `n` correctly rounded to `f64`?
pub fn is_correctly_rounded_f64(n: &BigInt, r: f64) -> bool {
    is_nearest_even(
        n,
        r,
        1024,
        [r.next_up(), r.next_down()],
        r.to_bits() & 1 == 0,
    )
}

/// Is `r` the exact sum `n` correctly rounded to `f32`?
pub fn is_correctly_rounded_f32(n: &BigInt, r: f32) -> bool {
    is_nearest_even(
        n,
        f64::from(r),
        128,
        [f64::from(r.next_up()), f64::from(r.next_down())],
        r.to_bits() & 1 == 0,
    )
}

/// The result IEEE-754 semantics prescribe if specials are present.
pub fn expected_special(values: &[f64]) -> Option<f64> {
    let has_pos = values.contains(&f64::INFINITY);
    let has_neg = values.contains(&f64::NEG_INFINITY);
    if values.iter().any(|v| v.is_nan()) || (has_pos && has_neg) {
        Some(f64::NAN)
    } else if has_pos {
        Some(f64::INFINITY)
    } else if has_neg {
        Some(f64::NEG_INFINITY)
    } else {
        None
    }
}

fn small(values: &[f64]) -> SmallAccumulator {
    let mut acc = SmallAccumulator::new();
    acc.add_slice(values);
    acc
}

fn large<const LANES: usize>(values: &[f64]) -> LargeAccumulator<LANES> {
    let mut acc = LargeAccumulator::<LANES>::new();
    acc.add_slice(values);
    acc
}

/// Sums `values` with every implementation variant.
pub fn all_sums(values: &[f64]) -> Vec<(&'static str, f64)> {
    let mut merged = small(&values[..values.len() / 3]);
    merged.merge(&large::<2>(&values[values.len() / 3..]).to_small());
    let one_by_one: SmallAccumulator = values.iter().copied().collect();
    vec![
        ("sum_f64", xsum_core::sum_f64(values)),
        ("par_sum_f64", xsum_core::par_sum_f64(values)),
        ("small", small(values).to_f64()),
        ("small one by one", one_by_one.to_f64()),
        ("large<1>", large::<1>(values).to_f64()),
        ("large<2>", large::<2>(values).to_f64()),
        ("large<3>", large::<3>(values).to_f64()),
        ("large<4>", large::<4>(values).to_f64()),
        ("large<8>", large::<8>(values).to_f64()),
        ("merged", merged.to_f64()),
    ]
}

/// Sums `values` with every `f32` implementation variant.
pub fn all_sums_f32(values: &[f32]) -> Vec<(&'static str, f32)> {
    let mut small_acc = SmallAccumulator::new();
    small_acc.add_slice(values);
    let mut large_acc = LargeAccumulator::<4>::new();
    large_acc.add_slice(values);
    vec![
        ("sum_f32", xsum_core::sum_f32(values)),
        ("par_sum_f32", xsum_core::par_sum_f32(values)),
        ("small", small_acc.to_f32()),
        ("large<4>", large_acc.to_f32()),
    ]
}

/// Asserts that all variants agree bit-for-bit and are correctly rounded (or
/// produce the prescribed special value).
pub fn assert_exact(values: &[f64]) {
    let sums = all_sums(values);
    let (_, first) = sums[0];
    for &(name, result) in &sums {
        assert!(
            result.to_bits() == first.to_bits() || (result.is_nan() && first.is_nan()),
            "{name} gave {result:e}, sum_f64 gave {first:e}"
        );
    }
    match expected_special(values) {
        Some(special) if special.is_nan() => assert!(first.is_nan(), "expected NaN, got {first}"),
        Some(special) => assert_eq!(first, special),
        None => assert!(
            is_correctly_rounded_f64(&exact_sum(values), first),
            "{first:e} is not the correctly rounded sum"
        ),
    }
}

/// `f32` counterpart of [`assert_exact`].
pub fn assert_exact_f32(values: &[f32]) {
    let sums = all_sums_f32(values);
    let (_, first) = sums[0];
    for &(name, result) in &sums {
        assert!(
            result.to_bits() == first.to_bits() || (result.is_nan() && first.is_nan()),
            "{name} gave {result:e}, sum_f32 gave {first:e}"
        );
    }
    let wide: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    match expected_special(&wide) {
        Some(special) if special.is_nan() => assert!(first.is_nan(), "expected NaN, got {first}"),
        Some(special) => assert_eq!(f64::from(first), special),
        None => assert!(
            is_correctly_rounded_f32(&exact_sum(&wide), first),
            "{first:e} is not the correctly rounded sum"
        ),
    }
}
