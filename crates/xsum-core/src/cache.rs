//! A per-thread cache of large accumulators.
//!
//! A fresh [`LargeAccumulator`] is a zeroed allocation of about 160 KiB. Its
//! pages are faulted in on first touch, which can cost more than summing tens
//! of thousands of values, especially when the exponents are widely spread.
//! Cleared accumulators are therefore kept per thread and reused. Clearing
//! costs only as much as the chunks that were used.

use std::cell::RefCell;

use crate::{LargeAccumulator, SmallAccumulator};

/// Upper bound on the accumulators kept per thread. A thread may need more
/// than one at a time when rayon runs a stolen task while it waits.
const MAX_CACHED_PER_THREAD: usize = 2;

thread_local! {
    static CACHE: RefCell<Vec<LargeAccumulator>> = const { RefCell::new(Vec::new()) };
}

/// Returns an empty large accumulator, reusing a cached one if available.
pub(crate) fn take() -> LargeAccumulator {
    CACHE.with_borrow_mut(Vec::pop).unwrap_or_default()
}

/// Returns the sum held by `acc` and puts the cleared accumulator back into the
/// cache.
pub(crate) fn finish(mut acc: LargeAccumulator) -> SmallAccumulator {
    let sum = acc.to_small();
    acc.clear();
    CACHE.with_borrow_mut(|cache| {
        if cache.len() < MAX_CACHED_PER_THREAD {
            cache.push(acc);
        }
    });
    sum
}

/// Sums the values that `feed` adds to a [`LargeAccumulator`], and returns the
/// exact sum.
///
/// The accumulator comes from a per-thread cache, so repeated calls avoid the
/// cost of allocating and first touching a large accumulator.
///
/// # Examples
///
/// ```
/// let parts: [&[f64]; 2] = [&[1e300, 0.5], &[-1e300, 0.25]];
/// let sum = xsum_core::sum_with(|acc| {
///     for part in parts {
///         acc.add_slice(part);
///     }
/// });
/// assert_eq!(sum.to_f64(), 0.75);
/// ```
pub fn sum_with<F: FnOnce(&mut LargeAccumulator)>(feed: F) -> SmallAccumulator {
    let mut acc = take();
    feed(&mut acc);
    finish(acc)
}
