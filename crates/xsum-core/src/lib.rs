//! Exact, correctly rounded summation of floating-point numbers.
//!
//! This crate implements the *superaccumulators* of Radford M. Neal,
//! [*Fast exact summation using small and large superaccumulators*](https://arxiv.org/abs/1505.05571)
//! (2015). A superaccumulator holds the **exact** sum of any number of `f64`
//! values as a long fixed-point number covering the whole `f64` exponent
//! range. Only the final result is rounded, exactly once.
//!
//! The result is therefore
//!
//! * **correctly rounded**: the exact sum rounded to nearest, ties to even;
//! * **independent of the summation order**, the chunking and the number of
//!   threads;
//! * **free of spurious overflow**: `[f64::MAX, f64::MAX, -f64::MAX]` sums to
//!   `f64::MAX`.
//!
//! # Special values
//!
//! * Any NaN summand gives NaN.
//! * Infinities of both signs give NaN.
//! * Otherwise an infinite summand gives that infinity.
//! * A finite exact sum beyond the range of the result type gives `±∞`.
//! * An exact zero (including an empty sum) gives `+0.0`.
//!
//! # Choosing an API
//!
//! * [`sum_f64`] / [`sum_f32`] sum a slice on the current thread. They use a
//!   [`SmallAccumulator`] for short inputs and a [`LargeAccumulator`] otherwise.
//! * [`par_sum_f64`] / [`par_sum_f32`] (feature `parallel`, enabled by
//!   default) additionally spread long inputs over the rayon thread pool.
//! * [`sum_with`] sums data that is not a single slice (e.g. several chunks)
//!   with a reused, thread-local [`LargeAccumulator`].
//! * The accumulators can be used directly for streaming input. Partial sums
//!   can be combined with [`SmallAccumulator::merge`].
//!
//! # Examples
//!
//! ```
//! let values = [1e100, 1.0, -1e100];
//! assert_eq!(values.iter().sum::<f64>(), 0.0); // naive summation loses the 1.0
//! assert_eq!(xsum_core::sum_f64(&values), 1.0);
//!
//! let values = [f64::MAX, f64::MAX, -f64::MAX];
//! assert_eq!(values.iter().sum::<f64>(), f64::INFINITY);
//! assert_eq!(xsum_core::sum_f64(&values), f64::MAX);
//! ```

#![forbid(unsafe_code)]

mod cache;
mod ieee;
mod large;
#[cfg(feature = "parallel")]
mod parallel;
mod round;
mod small;

pub use cache::sum_with;
pub use large::{DEFAULT_LANES, LargeAccumulator};
#[cfg(feature = "parallel")]
pub use parallel::{PARALLEL_THRESHOLD, block_len, par_sum_blocks, par_sum_f32, par_sum_f64};
pub use small::SmallAccumulator;

/// Inputs shorter than this are summed with a [`SmallAccumulator`], longer
/// ones with a [`LargeAccumulator`] (crossover determined by benchmarking).
pub const SMALL_THRESHOLD: usize = 1536;

/// Returns the exact sum of `values` correctly rounded to `f64`.
///
/// # Examples
///
/// ```
/// assert_eq!(xsum_core::sum_f64(&[0.1; 10]), 1.0);
/// assert_eq!(xsum_core::sum_f64(&[]), 0.0);
/// assert!(xsum_core::sum_f64(&[f64::INFINITY, f64::NEG_INFINITY]).is_nan());
/// ```
#[must_use]
pub fn sum_f64(values: &[f64]) -> f64 {
    exact_sum(values).to_f64()
}

/// Returns the exact sum of `values` correctly rounded to `f32`.
///
/// No precision is lost in intermediate steps. In particular, there is no
/// double rounding via `f64`.
///
/// # Examples
///
/// ```
/// let values = [1.0_f32, 2f32.powi(-24), 2f32.powi(-60)];
/// // Rounding the exact sum to f64 first and then to f32 would give 1.0.
/// assert_eq!(xsum_core::sum_f32(&values), 1.0 + f32::EPSILON);
/// ```
#[must_use]
pub fn sum_f32(values: &[f32]) -> f32 {
    exact_sum(values).to_f32()
}

/// Sums `values` exactly on the current thread, picking the accumulator by
/// input length.
pub(crate) fn exact_sum<T: Copy + Into<f64>>(values: &[T]) -> SmallAccumulator {
    if values.len() < SMALL_THRESHOLD {
        let mut acc = SmallAccumulator::new();
        acc.add_slice(values);
        acc
    } else {
        sum_with(|acc| acc.add_slice(values))
    }
}
