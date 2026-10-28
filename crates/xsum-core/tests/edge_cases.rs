//! Hand-picked edge cases: special values, overflow, underflow, ties,
//! cancellation and long inputs.

mod common;

use common::{assert_exact, assert_exact_f32, exact_sum, is_correctly_rounded_f64};
use xsum_core::{PARALLEL_THRESHOLD, SMALL_THRESHOLD, par_sum_f64, sum_f32, sum_f64};

const HALF_ULP_OF_MAX: f64 = 9.979_201_547_673_6e291; // 2^970

#[test]
fn empty_sum_is_positive_zero() {
    assert_eq!(sum_f64(&[]).to_bits(), 0);
    assert_eq!(sum_f32(&[]).to_bits(), 0);
    assert_exact(&[]);
}

#[test]
fn zeros_sum_to_positive_zero() {
    assert_eq!(sum_f64(&[-0.0]).to_bits(), 0);
    assert_eq!(sum_f64(&[-0.0, -0.0, 0.0]).to_bits(), 0);
    assert_eq!(sum_f64(&[1.5, -1.5]).to_bits(), 0);
    assert_eq!(sum_f64(&vec![-0.0; 5000]).to_bits(), 0);
    assert_eq!(sum_f32(&[-0.0, -0.0]).to_bits(), 0);
}

#[test]
fn nan_propagates() {
    for values in [
        vec![f64::NAN],
        vec![1.0, f64::NAN, 2.0],
        vec![f64::NAN, f64::INFINITY],
        vec![f64::NEG_INFINITY, -f64::NAN],
        vec![f64::INFINITY, f64::NEG_INFINITY],
        vec![f64::NEG_INFINITY, 1.0, f64::INFINITY],
    ] {
        assert!(sum_f64(&values).is_nan(), "{values:?}");
        assert_exact(&values);
    }
    assert!(sum_f32(&[1.0, f32::NAN]).is_nan());
    assert!(sum_f32(&[f32::INFINITY, f32::NEG_INFINITY]).is_nan());
}

#[test]
fn nan_payload_of_first_nan_is_kept() {
    let payload = f64::from_bits(f64::NAN.to_bits() | 0xBEEF);
    let other = f64::from_bits(f64::NAN.to_bits() | 0x1234);
    let mut values = vec![1.0; 3 * SMALL_THRESHOLD];
    values[5] = payload;
    values[600] = other;
    assert_eq!(sum_f64(&values).to_bits(), payload.to_bits());
    assert_eq!(sum_f64(&values[..10]).to_bits(), payload.to_bits());
}

#[test]
fn infinities() {
    assert_eq!(sum_f64(&[f64::INFINITY, 1.0]), f64::INFINITY);
    assert_eq!(sum_f64(&[f64::INFINITY, f64::INFINITY]), f64::INFINITY);
    assert_eq!(
        sum_f64(&[-1.0, f64::NEG_INFINITY, -f64::MAX]),
        f64::NEG_INFINITY
    );
    // An infinity is not cancelled by finite values of any magnitude.
    assert_eq!(
        sum_f64(&[f64::INFINITY, -f64::MAX, -f64::MAX]),
        f64::INFINITY
    );
    assert_eq!(sum_f32(&[f32::NEG_INFINITY, f32::MAX]), f32::NEG_INFINITY);
}

#[test]
fn overflow_of_the_exact_sum() {
    assert_eq!(sum_f64(&[f64::MAX, f64::MAX]), f64::INFINITY);
    assert_eq!(sum_f64(&[-f64::MAX, -f64::MAX]), f64::NEG_INFINITY);
    assert_eq!(sum_f64(&vec![f64::MAX; 10_000]), f64::INFINITY);
    assert_eq!(sum_f32(&[f32::MAX, f32::MAX]), f32::INFINITY);
    assert_exact(&[f64::MAX, f64::MAX, 1.0]);
}

#[test]
fn intermediate_overflow_is_harmless() {
    assert_eq!(sum_f64(&[f64::MAX, f64::MAX, -f64::MAX]), f64::MAX);
    assert_eq!(sum_f64(&[-f64::MAX, -f64::MAX, f64::MAX, 1.0]), -f64::MAX);
    let mut values = vec![f64::MAX; 5000];
    values.extend(vec![-f64::MAX; 5000]);
    values.push(42.0);
    assert_eq!(sum_f64(&values), 42.0);
    assert_eq!(par_sum_f64(&values), 42.0);
    // f32 sums never overflow internally either.
    assert_eq!(sum_f32(&[f32::MAX, f32::MAX, -f32::MAX]), f32::MAX);
}

#[test]
fn rounding_at_the_overflow_threshold() {
    // MAX + half an ulp is a tie; MAX has an odd significand, so it rounds up
    // to infinity, just like IEEE addition does.
    assert_eq!(HALF_ULP_OF_MAX, 2f64.powi(970));
    assert_eq!(sum_f64(&[f64::MAX, HALF_ULP_OF_MAX]), f64::INFINITY);
    assert_eq!(sum_f64(&[-f64::MAX, -HALF_ULP_OF_MAX]), f64::NEG_INFINITY);
    // Just below the tie stays finite.
    let below = [f64::MAX, HALF_ULP_OF_MAX, -f64::from_bits(1)];
    assert_eq!(sum_f64(&below), f64::MAX);
    assert_exact(&below);
    // f32: MAX + half an ulp (2^103) overflows, slightly less does not.
    assert_eq!(sum_f32(&[f32::MAX, 2f32.powi(103)]), f32::INFINITY);
    assert_eq!(
        sum_f32(&[f32::MAX, 2f32.powi(103), -f32::from_bits(1)]),
        f32::MAX
    );
}

#[test]
fn subnormals() {
    let tiny = f64::from_bits(1);
    assert_eq!(sum_f64(&[tiny; 7]), f64::from_bits(7));
    assert_eq!(sum_f64(&[tiny, -tiny]).to_bits(), 0);
    // Subnormals summing into the normal range.
    let largest_subnormal = f64::from_bits((1 << 52) - 1);
    assert_eq!(sum_f64(&[largest_subnormal, tiny]), f64::MIN_POSITIVE);
    let values = vec![largest_subnormal; 3000];
    assert_exact(&values);
    // Normal values cancelling down to a subnormal.
    assert_eq!(sum_f64(&[f64::MIN_POSITIVE, -largest_subnormal]), tiny);
    assert_eq!(sum_f32(&[f32::from_bits(1); 3]), f32::from_bits(3));
}

#[test]
fn underflow_results_are_exact() {
    // The exact sum of normal numbers can be tiny; it is still exact.
    let a = 1.0 + f64::EPSILON;
    assert_eq!(sum_f64(&[a, -1.0, -f64::EPSILON, 1e-300]), 1e-300);
    assert_eq!(sum_f64(&[1e300, 5e-324, -1e300]), 5e-324);
}

#[test]
fn catastrophic_cancellation() {
    assert_eq!(sum_f64(&[1.0, 1e100, 1.0, -1e100]), 2.0);
    assert_eq!(sum_f64(&[1e16, 1.0, -1e16]), 1.0);
    assert_eq!(sum_f64(&[1e308, 1.0, -1e308, 1e-308]), 1.0);
    assert_exact(&[3.0, 1e308, -1e308, 2f64.powi(-1074)]);
}

#[test]
fn naive_summation_is_inexact_but_xsum_is_not() {
    assert_eq!(sum_f64(&[0.1; 10]), 1.0);
    let values = [0.1, 0.2, 0.3];
    // Naive left-to-right summation rounds twice and ends one ulp too high.
    assert_eq!(values.iter().sum::<f64>(), 0.600_000_000_000_000_1);
    assert_eq!(sum_f64(&values), 0.6);
    assert_exact(&values);
}

#[test]
fn ties_round_to_even() {
    let eps = f64::EPSILON;
    // 1 + eps/2 is a tie between 1 and 1 + eps: round to even (1).
    assert_eq!(sum_f64(&[1.0, eps / 2.0]), 1.0);
    // ... but any extra amount breaks the tie.
    assert_eq!(sum_f64(&[1.0, eps / 2.0, 1e-300]), 1.0 + eps);
    // 1 + 3eps/2 is a tie between 1+eps and 1+2eps: round to even (1+2eps).
    assert_eq!(sum_f64(&[1.0, eps, eps / 2.0]), 1.0 + 2.0 * eps);
    assert_eq!(sum_f64(&[-1.0, -eps, -eps / 2.0]), -1.0 - 2.0 * eps);
}

#[test]
fn f32_has_no_double_rounding() {
    // Exact: 1 + 2^-24 + 2^-60, just above the half-way point between 1 and
    // 1 + 2^-23. Rounding to f64 first would drop the 2^-60 and create a tie
    // that then rounds to 1.0.
    let values = [1.0_f32, 2f32.powi(-24), 2f32.powi(-60)];
    assert_eq!(sum_f32(&values), 1.0 + f32::EPSILON);
    let wide: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
    assert_eq!(
        sum_f64(&wide) as f32,
        1.0,
        "the naive approach double-rounds"
    );
    assert_exact_f32(&values);
    assert_exact_f32(&[1.0, 2f32.powi(-24)]); // exact tie: round to even
}

#[test]
fn transitions_between_accumulators() {
    for n in [
        SMALL_THRESHOLD - 1,
        SMALL_THRESHOLD,
        SMALL_THRESHOLD + 1,
        PARALLEL_THRESHOLD - 1,
        PARALLEL_THRESHOLD,
        PARALLEL_THRESHOLD + 1,
    ] {
        let values: Vec<f64> = (0..n)
            .map(|i| (i as f64 * 0.754_877_666_246_692_7).fract() - 0.5)
            .collect();
        assert_exact(&values);
    }
}

#[test]
fn long_input_with_known_sum() {
    // Each triple sums to exactly 1, but naive summation loses every 1.0
    // because the spacing of doubles near 1e16 is 2.
    let triples = 3_333_333;
    let values: Vec<f64> = [1e16, 1.0, -1e16].repeat(triples);
    assert_eq!(values.iter().sum::<f64>(), 0.0);
    assert_eq!(par_sum_f64(&values), triples as f64);
    assert_eq!(sum_f64(&values), triples as f64);
}

#[test]
fn long_random_input_matches_reference() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let values: Vec<f64> = (0..1_000_000)
        .map(|_| {
            // xorshift64*: random finite bit patterns over the whole range.
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let value = f64::from_bits(bits);
            if value.is_finite() { value } else { 1.0 }
        })
        .collect();
    let result = par_sum_f64(&values);
    assert_eq!(result.to_bits(), sum_f64(&values).to_bits());
    assert!(is_correctly_rounded_f64(&exact_sum(&values), result));
}
