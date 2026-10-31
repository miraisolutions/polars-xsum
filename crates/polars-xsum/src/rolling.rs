//! Exact rolling sums: over windows of a fixed number of rows
//! (`rolling_xsum`) and over windows of a key column (`rolling_xsum_by`).
//!
//! The window slides over the column with a single small accumulator: a value
//! entering the window is added, a value leaving it is added negated. Both are
//! exact, so every window sum is exact and the cost per row depends only on how
//! far the window moves, not on its size. Infinities and NaNs cannot be removed
//! that way; they are counted separately.

use std::collections::VecDeque;

use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};
use polars_arrow::types::NativeType;
use rayon::prelude::*;
use serde::Deserialize;
use xsum_core::SmallAccumulator;

use crate::pool;

/// Minimum number of rows per parallel task.
const MIN_BLOCK_ROWS: usize = 1 << 14;

/// The shape of the windows, with the semantics of Polars' `rolling_sum`.
///
/// These are the keyword arguments of `rolling_xsum`.
#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct Window {
    /// Number of rows per window.
    #[serde(rename = "window_size")]
    pub(crate) size: usize,
    /// Number of non-null values a window needs for a non-null sum.
    pub(crate) min_samples: usize,
    /// Whether the window is centred on its row instead of ending at it.
    pub(crate) center: bool,
}

impl Window {
    /// The rows `start..end` of the window of `row`, clipped to `0..len`.
    fn bounds(self, row: usize, len: usize) -> (usize, usize) {
        let (start, end) = if self.center {
            let before = self.size / 2;
            (row.saturating_sub(before), row + (self.size - before))
        } else {
            ((row + 1).saturating_sub(self.size), row + 1)
        };
        (start, end.min(len))
    }
}

/// Which ends of the interval `[lower, key]` belong to a window.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Closed {
    /// Only the lower end: `[lower, key)`.
    Left,
    /// Only the upper end: `(lower, key]`.
    Right,
    /// Both ends: `[lower, key]`.
    Both,
    /// Neither end: `(lower, key)`.
    None,
}

/// The windows of `rolling_xsum_by`, with the semantics of Polars'
/// `rolling_sum_by`: the window of a row holds the rows whose key lies in
/// `[lower, key]` of that row, with ends included as given by `closed`.
///
/// These are the keyword arguments of `rolling_xsum_by`.
#[derive(Clone, Copy, Debug, Deserialize)]
pub(crate) struct KeyWindow {
    /// Number of non-null values a window needs for a non-null sum.
    pub(crate) min_samples: usize,
    /// Which ends of `[lower, key]` belong to the window.
    pub(crate) closed: Closed,
    /// The window size of an integer key column (`"3i"` gives 3), so that
    /// `lower = key - window_size`. `None` for a duration on a temporal key
    /// column: Polars then computes the lower ends, which are passed as a
    /// column.
    pub(crate) window_size: Option<u64>,
}

/// Sums every window of rows of `ca` exactly, as `rolling_xsum` does.
///
/// # Arguments
///
/// * `ca` - The values. Nulls are skipped.
/// * `window` - The size and position of the windows, and the number of
///   non-null values a window needs.
/// * `round` - Rounds an exact sum to `T` (`SmallAccumulator::to_f32` or
///   `SmallAccumulator::to_f64`).
///
/// # Returns
///
/// One sum per row of `ca`, with the name of `ca`. A window with fewer than
/// `window.min_samples` non-null values gives null.
///
/// # Errors
///
/// Fails if `window.min_samples` exceeds `window.size`.
pub(crate) fn exact_rolling_sums<T>(
    ca: &ChunkedArray<T>,
    window: Window,
    round: impl Fn(&SmallAccumulator) -> T::Native + Sync,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Into<f64>,
{
    polars_ensure!(
        window.min_samples <= window.size,
        InvalidOperation: "min_samples ({}) must be <= window_size ({})",
        window.min_samples,
        window.size
    );
    let ca = ca.rechunk();
    let len = ca.len();
    let Some(array) = ca.downcast_iter().next() else {
        return Ok(ca.into_owned());
    };
    let sums = slide(
        array,
        |row| window.bounds(row, len),
        window.size,
        window.min_samples,
        &round,
    );
    Ok(ChunkedArray::from_iter_options(
        ca.name().clone(),
        sums.into_iter(),
    ))
}

/// Sums every window of `ca` over a key column exactly, as `rolling_xsum_by`
/// does.
///
/// The rows are sorted by key unless they already are. The window slides over
/// them in that order, and the sums are put back into the original row order.
///
/// # Arguments
///
/// * `ca` - The values. Nulls are skipped.
/// * `keys` - The physical (integer) values of the key column, which need not
///   be sorted. A row with a null key belongs to no window.
/// * `lower` - The physical values of each row's lower window end, as computed
///   by Polars for a duration window. Used only when `window.window_size` is
///   `None`.
/// * `window` - Which ends of the window interval are included, the window
///   size of an integer key column, and the number of non-null values a
///   window needs.
/// * `round` - Rounds an exact sum to `T` (`SmallAccumulator::to_f32` or
///   `SmallAccumulator::to_f64`).
///
/// # Returns
///
/// One sum per row of `ca`, in its original order and with its name. A row
/// gives null if its key or its lower end is null, or if its window has fewer
/// than `window.min_samples` non-null values.
///
/// # Errors
///
/// Fails if `keys` or `lower` differ in length from `ca`.
pub(crate) fn exact_rolling_sums_by<T>(
    ca: &ChunkedArray<T>,
    keys: &Int64Chunked,
    lower: Option<&Int64Chunked>,
    window: KeyWindow,
    round: impl Fn(&SmallAccumulator) -> T::Native + Sync,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsFloatType,
    T::Native: Into<f64>,
{
    let len = ca.len();
    polars_ensure!(
        keys.len() == len,
        ShapeMismatch: "rolling_xsum_by: the values ({len}) and the key column ({}) must have \
                        the same length",
        keys.len()
    );
    if let Some(lower) = lower {
        polars_ensure!(
            lower.len() == len,
            ShapeMismatch: "rolling_xsum_by: the values ({len}) and the lower window ends ({}) \
                            must have the same length",
            lower.len()
        );
    }
    let (ca, keys) = (ca.rechunk(), keys.rechunk());
    let lower = lower.map(ChunkedArray::rechunk);
    let (Some(array), Some(keys)) = (ca.downcast_iter().next(), keys.downcast_iter().next()) else {
        return Ok(ca.into_owned());
    };
    let lower = lower
        .as_ref()
        .and_then(|lower| lower.downcast_iter().next());
    // The lower window end of `row`, `None` if it is null. It is wider than
    // the keys, so that `key - size` cannot overflow.
    let lower_end = |row: usize| match (window.window_size, lower) {
        (Some(size), _) => Some(i128::from(keys.value(row)) - i128::from(size)),
        (None, Some(lower)) => lower.get(row).map(i128::from),
        (None, None) => None,
    };
    // Small inputs, e.g. the groups of an `over`, stay on the calling thread.
    let parallel = len >= MIN_BLOCK_ROWS;
    // The rows with a key, as `(key, row)`, sorted by key and then by row.
    let mut sorted: Vec<(i64, usize)> = (0..len)
        .filter(|&row| keys.is_valid(row))
        .map(|row| (keys.value(row), row))
        .collect();
    let in_order = sorted.is_sorted();
    if !in_order && parallel {
        pool::install(|| sorted.par_sort_unstable());
    } else if !in_order {
        sorted.sort_unstable();
    }
    // Without null keys, rows already sorted by key need no reordering.
    let identity = in_order && sorted.len() == len;
    let gathered: PrimitiveArray<T::Native>;
    let values = if identity {
        array
    } else {
        gathered = sorted.iter().map(|&(_, row)| array.get(row)).collect();
        &gathered
    };

    // The window of the row at sorted position `k`, as sorted positions.
    let (include_low, include_high) = match window.closed {
        Closed::Left => (true, false),
        Closed::Right => (false, true),
        Closed::Both => (true, true),
        Closed::None => (false, false),
    };
    // A row without a lower end gets an empty window and, below, a null sum.
    let bounds = |k: usize| {
        let (high, row) = sorted[k];
        let Some(low) = lower_end(row) else {
            return (k, k);
        };
        // A lower end below every possible key starts the window at the first row.
        let start = i64::try_from(low).map_or(0, |low| {
            sorted.partition_point(|&(t, _)| t < low || (t == low && !include_low))
        });
        let end = sorted.partition_point(|&(t, _)| t < high || (t == high && include_high));
        (start, end.max(start))
    };
    let all_bounds: Vec<(usize, usize)> = if parallel {
        pool::install(|| (0..sorted.len()).into_par_iter().map(bounds).collect())
    } else {
        (0..sorted.len()).map(bounds).collect()
    };
    let longest = all_bounds.iter().map(|(start, end)| end - start).max();
    let mut sums = slide(
        values,
        |k| all_bounds[k],
        longest.unwrap_or(0),
        window.min_samples,
        &round,
    );
    for (sum, &(_, row)) in sums.iter_mut().zip(&sorted) {
        if lower_end(row).is_none() {
            *sum = None;
        }
    }
    if identity {
        return Ok(ChunkedArray::from_iter_options(
            ca.name().clone(),
            sums.into_iter(),
        ));
    }
    let mut out = vec![None; len];
    for ((_, row), sum) in sorted.into_iter().zip(sums) {
        out[row] = sum;
    }
    Ok(ChunkedArray::from_iter_options(
        ca.name().clone(),
        out.into_iter(),
    ))
}

/// Sums the window of every row of `array` exactly, sliding one accumulator
/// over the rows.
///
/// Long columns are split into blocks of rows that are summed in parallel on
/// the plugin's thread pool. Each block first sums the full window of its first
/// row from scratch. Blocks are at least as long as the longest window, which
/// bounds that extra work to at most doubling the total.
///
/// # Arguments
///
/// * `array` - The values, in the order in which the window slides over them.
/// * `bounds` - The window of a row, as the rows `start..end` of `array`. Both
///   ends may move in either direction from one row to the next; the cost
///   grows with how far they move.
/// * `longest` - The length of the longest window. Blocks are at least this
///   long.
/// * `min_samples` - The number of non-null values a window needs for a
///   non-null sum.
/// * `round` - Rounds an exact sum to the output type.
///
/// # Returns
///
/// One sum per row of `array`, in row order: `None` where the window has
/// fewer than `min_samples` non-null values.
fn slide<T, N>(
    array: &PrimitiveArray<T>,
    bounds: impl Fn(usize) -> (usize, usize) + Sync,
    longest: usize,
    min_samples: usize,
    round: &(impl Fn(&SmallAccumulator) -> N + Sync),
) -> Vec<Option<N>>
where
    T: NativeType + Into<f64>,
    N: Send,
{
    let len = array.len();
    if len == 0 {
        return Vec::new();
    }
    let block = (len / (4 * pool::size())).max(longest).max(MIN_BLOCK_ROWS);
    let sum_block = |first: usize| -> Vec<Option<N>> {
        let mut sliding = Sliding::new(array, bounds(first).0);
        (first..len.min(first + block))
            .map(|row| {
                let (start, end) = bounds(row);
                sliding.move_to(start, end);
                sliding.sum(min_samples, round)
            })
            .collect()
    };
    if len > block {
        let blocks: Vec<_> = pool::install(|| {
            (0..len)
                .into_par_iter()
                .step_by(block)
                .map(sum_block)
                .collect()
        });
        blocks.into_iter().flatten().collect()
    } else {
        sum_block(0)
    }
}

/// Whether a row enters or leaves the window.
#[derive(Clone, Copy)]
enum Step {
    /// The row joins the window; its value is added.
    Enter,
    /// The row leaves the window; its value is subtracted.
    Leave,
}

/// Which end of the window a row enters or leaves by.
#[derive(Clone, Copy)]
enum End {
    /// The lower end, `lo`.
    Low,
    /// The upper end, `hi`.
    High,
}

/// The exact sum of the rows `lo..hi` of an array, for a window that moves in
/// either direction.
struct Sliding<'a, T: NativeType> {
    /// The values the window slides over.
    array: &'a PrimitiveArray<T>,
    /// First row of the window.
    lo: usize,
    /// One past the last row of the window.
    hi: usize,
    /// Sum of the finite values in the window.
    acc: SmallAccumulator,
    /// Number of non-null values in the window.
    count: usize,
    /// Rows of the NaNs in the window, in order.
    nans: VecDeque<usize>,
    /// Number of `+∞` values in the window.
    positive_infinities: usize,
    /// Number of `−∞` values in the window.
    negative_infinities: usize,
}

impl<'a, T: NativeType + Into<f64>> Sliding<'a, T> {
    /// An empty window over `array`, at row `start`.
    fn new(array: &'a PrimitiveArray<T>, start: usize) -> Self {
        Self {
            array,
            lo: start,
            hi: start,
            acc: SmallAccumulator::new(),
            count: 0,
            nans: VecDeque::new(),
            positive_infinities: 0,
            negative_infinities: 0,
        }
    }

    /// Moves the window to the rows `start..end` (`start <= end`).
    ///
    /// The window first grows to cover both the old and the new rows, then
    /// shrinks to the new ones, so it is never inverted.
    fn move_to(&mut self, start: usize, end: usize) {
        debug_assert!(start <= end);
        while self.hi < end {
            self.update(self.hi, Step::Enter, End::High);
            self.hi += 1;
        }
        while self.lo > start {
            self.lo -= 1;
            self.update(self.lo, Step::Enter, End::Low);
        }
        while self.hi > end {
            self.hi -= 1;
            self.update(self.hi, Step::Leave, End::High);
        }
        while self.lo < start {
            self.update(self.lo, Step::Leave, End::Low);
            self.lo += 1;
        }
    }

    /// Adds or removes the value of `row` at the window's `end`; nulls are
    /// skipped.
    fn update(&mut self, row: usize, step: Step, end: End) {
        if !self.array.is_valid(row) {
            return;
        }
        let value: f64 = self.array.value(row).into();
        let count = |count: &mut usize| match step {
            Step::Enter => *count += 1,
            Step::Leave => *count -= 1,
        };
        count(&mut self.count);
        if value.is_finite() {
            self.acc.add(match step {
                Step::Enter => value,
                Step::Leave => -value,
            });
        } else if value.is_nan() {
            match (step, end) {
                (Step::Enter, End::Low) => self.nans.push_front(row),
                (Step::Enter, End::High) => self.nans.push_back(row),
                (Step::Leave, End::Low) => {
                    self.nans.pop_front();
                }
                (Step::Leave, End::High) => {
                    self.nans.pop_back();
                }
            }
        } else if value > 0.0 {
            count(&mut self.positive_infinities);
        } else {
            count(&mut self.negative_infinities);
        }
    }

    /// The window's sum, rounded with `round`, or null with fewer than
    /// `min_samples` non-null values.
    ///
    /// Special values follow `xsum`: the first NaN wins, and infinities of
    /// both signs give NaN. "First" refers to the order the window slides in,
    /// which for `rolling_xsum_by` with unsorted keys is the key order, not
    /// the row order.
    fn sum<N>(&self, min_samples: usize, round: impl Fn(&SmallAccumulator) -> N) -> Option<N> {
        if self.count < min_samples {
            return None;
        }
        let special = match self.nans.front() {
            Some(&row) => Some(self.array.value(row).into()),
            None => match (self.positive_infinities > 0, self.negative_infinities > 0) {
                (true, true) => Some(f64::NAN),
                (true, false) => Some(f64::INFINITY),
                (false, true) => Some(f64::NEG_INFINITY),
                (false, false) => None,
            },
        };
        Some(match special {
            // A fresh accumulator rounds the special value exactly as `xsum`.
            Some(value) => round(&std::iter::once(value).collect()),
            None => round(&self.acc),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each window summed from scratch.
    fn expected(values: &[Option<f64>], window: Window) -> Vec<Option<u64>> {
        (0..values.len())
            .map(|row| {
                let (start, end) = window.bounds(row, values.len());
                let present: Vec<f64> = values[start..end].iter().flatten().copied().collect();
                (present.len() >= window.min_samples)
                    .then(|| xsum_core::sum_f64(&present).to_bits())
            })
            .collect()
    }

    fn rolling(values: &[Option<f64>], window: Window) -> Vec<Option<u64>> {
        let ca = Float64Chunked::from_iter_options("x".into(), values.iter().copied());
        exact_rolling_sums(&ca, window, SmallAccumulator::to_f64)
            .unwrap()
            .iter()
            .map(|sum| sum.map(f64::to_bits))
            .collect()
    }

    #[test]
    fn matches_windows_summed_from_scratch_across_blocks() {
        // Several blocks, ill-conditioned values, nulls and the odd special.
        let values: Vec<Option<f64>> = (0..2 * MIN_BLOCK_ROWS + 5)
            .map(|i| match i % 101 {
                0 => None,
                50 => Some(f64::INFINITY),
                77 if i % 3 == 0 => Some(f64::NAN),
                _ => Some(10f64.powi((i % 41) as i32 - 20) * (i as f64 * 0.37).sin()),
            })
            .collect();
        for size in [0, 1, 2, 7, 100] {
            for min_samples in [0, size / 2, size] {
                for center in [false, true] {
                    let window = Window {
                        size,
                        min_samples,
                        center,
                    };
                    assert_eq!(
                        rolling(&values, window),
                        expected(&values, window),
                        "{window:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn windows_may_move_backwards() {
        // Lower ends that jump back and forth, as calendar offsets might.
        let values: Vec<Option<f64>> = (0..60_u32)
            .map(|i| match i % 9 {
                0 => None,
                4 => Some(f64::from_bits(f64::NAN.to_bits() + u64::from(i))),
                _ => Some(10f64.powi(i as i32 % 30 - 15)),
            })
            .collect();
        let keys: Vec<i64> = (0..60).collect();
        let lower: Vec<i64> = (0..60).map(|k| k - (k * 7) % 13).collect();
        let window = KeyWindow {
            min_samples: 0,
            closed: Closed::Both,
            window_size: None,
        };
        let ca = Float64Chunked::from_iter_options("x".into(), values.iter().copied());
        let sums = exact_rolling_sums_by(
            &ca,
            &Int64Chunked::from_vec("t".into(), keys),
            Some(&Int64Chunked::from_vec("l".into(), lower.clone())),
            window,
            SmallAccumulator::to_f64,
        )
        .unwrap();
        for (row, sum) in sums.iter().enumerate() {
            let start = usize::try_from(lower[row].max(0)).unwrap();
            let present: Vec<f64> = values[start..=row].iter().flatten().copied().collect();
            assert_eq!(
                sum.map(f64::to_bits),
                Some(xsum_core::sum_f64(&present).to_bits()),
                "row {row}"
            );
        }
    }

    #[test]
    fn length_mismatches_name_the_offending_column() {
        let ca = Float64Chunked::from_slice("x".into(), &[1.0, 2.0]);
        let keys = Int64Chunked::from_slice("t".into(), &[1, 2]);
        let window = KeyWindow {
            min_samples: 0,
            closed: Closed::Right,
            window_size: None,
        };
        let error = |keys: &Int64Chunked, lower: &Int64Chunked| {
            exact_rolling_sums_by(&ca, keys, Some(lower), window, SmallAccumulator::to_f64)
                .unwrap_err()
                .to_string()
        };
        let short = Int64Chunked::from_slice("l".into(), &[0]);
        assert!(error(&short, &keys).contains("the key column (1)"));
        assert!(error(&keys, &short).contains("the lower window ends (1)"));
    }

    #[test]
    fn rejects_min_samples_above_window_size() {
        let ca = Float64Chunked::from_slice("x".into(), &[1.0]);
        let window = Window {
            size: 1,
            min_samples: 2,
            center: false,
        };
        assert!(exact_rolling_sums(&ca, window, SmallAccumulator::to_f64).is_err());
    }
}
