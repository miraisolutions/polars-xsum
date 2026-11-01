//! Multi-threaded summation with rayon.
//!
//! Superaccumulators can be merged exactly, so the input is split into blocks.
//! Every rayon task sums its blocks into its own [`LargeAccumulator`], and the
//! partial sums are merged as [`SmallAccumulator`]s. The result is bit-for-bit
//! identical to the sequential sum and does not depend on the number of threads.

use rayon::prelude::*;

use crate::{LargeAccumulator, SmallAccumulator, cache, exact_sum};

/// Inputs shorter than this are summed on the calling thread (crossover
/// determined by benchmarking).
pub const PARALLEL_THRESHOLD: usize = 1 << 15;

/// Lower bound for the number of elements per parallel work item (see
/// [`block_len`]). Smaller blocks balance the load better, larger ones reduce
/// scheduling overhead.
const MIN_BLOCK_LEN: usize = 1 << 12;

/// Upper bound for the number of elements per parallel work item (see
/// [`block_len`]). `2^14` performed best in benchmarks for inputs large enough
/// to use it.
const MAX_BLOCK_LEN: usize = 1 << 14;

/// Target number of rayon tasks per thread in [`par_sum_blocks`], and of
/// blocks per thread in [`block_len`].
pub const TASKS_PER_THREAD: usize = 4;

/// Returns the block length for summing `len` elements: about
/// [`TASKS_PER_THREAD`] blocks per thread, clamped to `[2^12, 2^14]`.
///
/// The threads are those of the current rayon thread pool. Outside any pool,
/// that is the global pool, which this starts if it is not running yet.
#[must_use]
pub fn block_len(len: usize) -> usize {
    (len / (TASKS_PER_THREAD * rayon::current_num_threads())).clamp(MIN_BLOCK_LEN, MAX_BLOCK_LEN)
}

/// Returns the exact sum of `values` correctly rounded to `f64`, using all
/// threads of the current rayon thread pool for long inputs.
///
/// The result is identical to [`sum_f64`](crate::sum_f64).
///
/// # Examples
///
/// ```
/// let values = [1e20, 1.0, -1e20].repeat(500_000);
/// assert_eq!(values.iter().sum::<f64>(), 0.0);
/// assert_eq!(xsum_core::par_sum_f64(&values), 500_000.0);
/// ```
#[must_use]
pub fn par_sum_f64(values: &[f64]) -> f64 {
    par_exact_sum(values).to_f64()
}

/// Returns the exact sum of `values` correctly rounded to `f32`, using all
/// threads of the current rayon thread pool for long inputs.
///
/// The result is identical to [`sum_f32`](crate::sum_f32).
#[must_use]
pub fn par_sum_f32(values: &[f32]) -> f32 {
    par_exact_sum(values).to_f32()
}

/// Returns the exact sum of `values`, computed in parallel if they are long
/// enough.
fn par_exact_sum<T: Copy + Into<f64> + Sync>(values: &[T]) -> SmallAccumulator {
    if values.len() < PARALLEL_THRESHOLD {
        return exact_sum(values);
    }
    par_sum_blocks(values.par_chunks(block_len(values.len())), |acc, block| {
        acc.add_slice(block);
    })
}

/// Sums the values of all work items exactly, in parallel.
///
/// Each rayon task owns one [`LargeAccumulator`] (taken from a per-thread
/// cache) and processes a run of consecutive items, about
/// [`TASKS_PER_THREAD`] tasks per thread. The partial sums are merged exactly at the end. The result does not
/// depend on how rayon splits the work.
///
/// This is the building block for summing data that is not a single slice,
/// e.g. the chunks of a columnar array that may contain missing values.
///
/// # Arguments
///
/// * `items` - The work items, e.g. the blocks of a slice or the chunks of an
///   array.
/// * `feed` - Adds the values of one work item to an accumulator. It is called
///   once per item, on any thread of the current rayon thread pool.
///
/// # Returns
///
/// The exact sum, unrounded: [`SmallAccumulator::to_f64`] or
/// [`SmallAccumulator::to_f32`] rounds it.
///
/// # Examples
///
/// ```
/// use rayon::prelude::*;
///
/// let chunks: Vec<Vec<f64>> = vec![vec![1e300, 1.0], vec![-1e300, 2.0]];
/// let acc = xsum_core::par_sum_blocks(chunks.par_iter(), |acc, chunk| acc.add_slice(chunk));
/// assert_eq!(acc.to_f64(), 3.0);
/// ```
pub fn par_sum_blocks<I, F>(items: I, feed: F) -> SmallAccumulator
where
    I: IndexedParallelIterator,
    F: Fn(&mut LargeAccumulator, I::Item) + Sync + Send,
{
    // Every task starts with an empty accumulator whose chunks must be
    // initialised on first use. With widely spread exponents that costs as much
    // as summing thousands of values, so tasks should not be too small.
    let min_items_per_task = items
        .len()
        .div_ceil(TASKS_PER_THREAD * rayon::current_num_threads());
    items
        .with_min_len(min_items_per_task.max(1))
        .fold(cache::take, |mut acc, item| {
            feed(&mut acc, item);
            acc
        })
        .map(cache::finish)
        .reduce(SmallAccumulator::new, |mut left, right| {
            left.merge(&right);
            left
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_equals_sequential() {
        let n = 3 * PARALLEL_THRESHOLD + 17;
        let values: Vec<f64> = (0..n)
            .map(|i| {
                let i = i as f64;
                (i * 0.618_033_988_749_895).fract()
                    * 10f64.powf((i % 37.0) - 18.0)
                    * if i % 3.0 == 0.0 { -1.0 } else { 1.0 }
            })
            .collect();
        let sequential = crate::sum_f64(&values);
        assert_eq!(par_sum_f64(&values).to_bits(), sequential.to_bits());

        let single: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        assert_eq!(
            par_sum_f32(&single).to_bits(),
            crate::sum_f32(&single).to_bits()
        );
    }

    #[test]
    fn parallel_specials() {
        let mut values = vec![1.0; 2 * PARALLEL_THRESHOLD];
        values[PARALLEL_THRESHOLD + 3] = f64::INFINITY;
        assert_eq!(par_sum_f64(&values), f64::INFINITY);
        values[7] = f64::NEG_INFINITY;
        assert!(par_sum_f64(&values).is_nan());
    }

    #[test]
    fn nan_payload_does_not_depend_on_thread_count() {
        // Opposite infinities in different blocks, followed by a NaN with a
        // payload: the payload must survive however the partial sums are merged.
        let payload = f64::from_bits(f64::NAN.to_bits() | 0xBEEF);
        let mut values = vec![1.0; 8 * MAX_BLOCK_LEN];
        values[10] = f64::INFINITY;
        values[3 * MAX_BLOCK_LEN] = f64::NEG_INFINITY;
        values[7 * MAX_BLOCK_LEN] = payload;
        assert_eq!(crate::sum_f64(&values).to_bits(), payload.to_bits());
        for threads in 1..=8 {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            for _ in 0..20 {
                let result = pool.install(|| par_sum_f64(&values));
                assert_eq!(result.to_bits(), payload.to_bits(), "{threads} threads");
            }
        }
    }
}
