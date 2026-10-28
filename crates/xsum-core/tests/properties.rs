//! Property-based tests against an exact arbitrary-precision reference.
//!
//! Every strategy generates inputs that stress a different part of the
//! algorithm: random bit patterns, clustered exponents (chunk rollover), wide
//! exponent spreads (carry propagation), cancellation, subnormals, values near
//! the overflow threshold and exact rounding ties.
//!
//! The number of cases can be raised with the `PROPTEST_CASES` environment
//! variable.

mod common;

use common::{assert_exact, assert_exact_f32};
use proptest::prelude::*;
use xsum_core::{SmallAccumulator, sum_f64};

/// Builds a finite double from its parts.
fn from_parts(negative: bool, biased_exp: u64, mantissa: u64) -> f64 {
    f64::from_bits(u64::from(negative) << 63 | biased_exp << 52 | mantissa)
}

/// Any bit pattern, including NaNs and infinities.
fn any_f64() -> impl Strategy<Value = f64> {
    prop_oneof![
        8 => any::<u64>().prop_map(f64::from_bits),
        1 => Just(f64::INFINITY),
        1 => Just(f64::NEG_INFINITY),
        1 => Just(f64::NAN),
    ]
}

/// Finite doubles from a mix of distributions.
fn finite_f64() -> impl Strategy<Value = f64> {
    let mantissa = 0..1_u64 << 52;
    prop_oneof![
        // Uniform over all finite bit patterns.
        (any::<bool>(), 0..=2046_u64, mantissa.clone()).prop_map(|(s, e, m)| from_parts(s, e, m)),
        // Clustered exponents: many values share a chunk.
        (any::<bool>(), 1020..=1026_u64, mantissa.clone())
            .prop_map(|(s, e, m)| from_parts(s, e, m)),
        // Subnormals and zeros.
        (any::<bool>(), mantissa.clone()).prop_map(|(s, m)| from_parts(s, 0, m)),
        // Near the overflow threshold.
        (any::<bool>(), 2040..=2046_u64, mantissa).prop_map(|(s, e, m)| from_parts(s, e, m)),
        // Ordinary numbers.
        -1e6..1e6_f64,
    ]
}

/// Values together with their negations (shuffled) and a few small extras,
/// so the exact sum is tiny compared to the summands.
fn cancelling() -> impl Strategy<Value = Vec<f64>> {
    (
        prop::collection::vec(finite_f64(), 0..400),
        prop::collection::vec(-1.0..1.0_f64, 0..4),
    )
        .prop_flat_map(|(values, extras)| {
            let mut all: Vec<f64> = values.iter().flat_map(|&v| [v, -v]).collect();
            all.extend(extras);
            Just(all).prop_shuffle()
        })
}

/// Sums that are exactly half-way between two doubles, optionally nudged.
fn ties() -> impl Strategy<Value = Vec<f64>> {
    (any::<bool>(), 1..=2045_u64, 0..1_u64 << 52, -1..=1_i32).prop_map(|(s, e, m, nudge)| {
        let base = from_parts(s, e, m);
        let half_ulp = (base.next_up() - base) / 2.0;
        let mut values = vec![base, half_ulp];
        if nudge != 0 {
            values.push(f64::from(nudge) * f64::from_bits(1));
        }
        values
    })
}

/// Arbitrary `f32` values (including specials).
fn any_f32() -> impl Strategy<Value = f32> {
    prop_oneof![
        8 => any::<u32>().prop_map(f32::from_bits),
        2 => -1e3..1e3_f32,
        1 => Just(f32::INFINITY),
        1 => Just(f32::NEG_INFINITY),
    ]
}

proptest! {
    #[test]
    fn random_bit_patterns(values in prop::collection::vec(any_f64(), 0..300)) {
        assert_exact(&values);
    }

    #[test]
    fn finite_mixtures(values in prop::collection::vec(finite_f64(), 0..300)) {
        assert_exact(&values);
    }

    #[test]
    fn cancellation(values in cancelling()) {
        assert_exact(&values);
    }

    #[test]
    fn exact_ties(values in ties()) {
        assert_exact(&values);
    }

    #[test]
    fn f32_inputs(values in prop::collection::vec(any_f32(), 0..300)) {
        assert_exact_f32(&values);
    }

    #[test]
    fn order_does_not_matter(values in prop::collection::vec(finite_f64(), 0..200).prop_shuffle()) {
        let mut reversed = values.clone();
        reversed.reverse();
        prop_assert_eq!(sum_f64(&values).to_bits(), sum_f64(&reversed).to_bits());
    }

    #[test]
    fn merging_any_partition(
        values in prop::collection::vec(finite_f64(), 0..300),
        cuts in prop::collection::vec(any::<prop::sample::Index>(), 0..5),
    ) {
        let mut cuts: Vec<usize> = cuts.iter().map(|c| c.index(values.len() + 1)).collect();
        cuts.sort_unstable();
        let mut merged = SmallAccumulator::new();
        let mut start = 0;
        for end in cuts.into_iter().chain([values.len()]) {
            let part: SmallAccumulator = values[start..end].iter().copied().collect();
            merged.merge(&part);
            start = end;
        }
        prop_assert_eq!(merged.to_f64().to_bits(), sum_f64(&values).to_bits());
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    /// Long inputs exercise chunk rollover and the large accumulator.
    #[test]
    fn long_inputs(values in prop::collection::vec(finite_f64(), 2000..20_000)) {
        assert_exact(&values);
    }

    /// Long runs of a single exponent stress the 2048-value chunk limit.
    #[test]
    fn long_runs_of_one_exponent(
        negative in any::<bool>(),
        exp in 1..=2046_u64,
        mantissas in prop::collection::vec(0..1_u64 << 52, 2000..12_000),
    ) {
        let values: Vec<f64> = mantissas.iter().map(|&m| from_parts(negative, exp, m)).collect();
        assert_exact(&values);
    }
}
