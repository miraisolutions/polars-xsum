//! The *large superaccumulator* (Neal 2015, section 4).
//!
//! There is one 64-bit chunk for every combination of sign and exponent, so
//! the chunk index is just the top 12 bits of a double's bit pattern. All
//! values in a chunk share sign and exponent, so a chunk can simply add up
//! **raw bit patterns**: after `n` additions it holds
//! `n · (index << 52) + Σ mantissa (mod 2^64)`. Since `n` is tracked as well,
//! the mantissa sum can be recovered exactly, as long as it cannot overflow.
//! That holds for `n ≤ 2048`. A full chunk is transferred to a
//! [`SmallAccumulator`] and starts again from scratch.
//!
//! `n` is stored as the remaining capacity `2048 − n` of a chunk, and unused
//! chunks have no capacity either. Decrementing the capacity borrows exactly
//! when the value needs the slow path, so the hot path is a single
//! decrement-and-branch plus a read-modify-write of the chunk.
//!
//! The hot path receives the tables as separate references. Through `&mut self`
//! the compiler would reload the table pointers after every store, as it cannot
//! rule out that a store into a table modifies the accumulator itself.
//!
//! # Lanes
//!
//! When consecutive values share an exponent (the common case), consecutive
//! additions update the same chunk and form a serial dependency chain through
//! memory. With `LANES > 1`, each index gets `LANES` independent chunks that
//! share a cache line, and consecutive values go to different lanes. The CPU
//! can then overlap the updates.

use crate::ieee::{EXP_MASK, IMPLICIT_BIT, MANTISSA_BITS, SIGN_BIT, is_special};
use crate::small::SmallAccumulator;

/// Number of chunks: one per combination of sign bit and 11 exponent bits.
const NUM_CHUNKS: usize = 1 << 12;

/// Number of 64-bit words of the bitmap of used chunk indices.
const USED_WORDS: usize = NUM_CHUNKS / 64;

/// Maximum number of values a chunk accumulates before being transferred.
const MAX_COUNT: u16 = 1 << (64 - (MANTISSA_BITS + 1));

/// The default number of lanes, chosen by benchmarking (see `BENCHMARKS.md`).
pub const DEFAULT_LANES: usize = 4;

const _: () = {
    // The sum of `MAX_COUNT` significands (implicit bit included) fits in u64.
    assert!(MAX_COUNT as u128 * ((IMPLICIT_BIT as u128) << 1) <= 1 << 64);
    assert!(USED_WORDS <= 64, "the summary bitmap is a single u64");
};

/// A table with `LANES` entries per chunk index.
type Table<T, const LANES: usize> = [[T; LANES]; NUM_CHUNKS];

/// An exact accumulator for sums of `f64` values optimised for throughput on
/// long inputs.
///
/// This accumulator holds about `40 · LANES` KiB of state. The state is zero
/// initialised, and only the chunks for exponents that occur are ever touched.
/// It falls back to an internal [`SmallAccumulator`] for the final carry
/// handling and for infinities and NaNs.
///
/// # Examples
///
/// ```
/// use xsum_core::LargeAccumulator;
///
/// let mut acc = LargeAccumulator::<4>::new();
/// acc.add_slice(&[1e100, 1.0, -1e100, 2.0_f64.powi(-60)]);
/// acc.add(1.0);
/// assert_eq!(acc.to_f64(), 2.0);
/// ```
#[derive(Clone, Debug)]
pub struct LargeAccumulator<const LANES: usize = DEFAULT_LANES> {
    /// Sums of raw bit patterns, per chunk index and lane.
    chunks: Box<Table<u64, LANES>>,
    /// Number of values each chunk can still take before it is full. All lanes
    /// of an unused index are 0, all lanes of a newly used one `MAX_COUNT`.
    remaining: Box<Table<u16, LANES>>,
    /// State that is only touched outside the hot path.
    cold: ColdState,
}

/// The parts of a [`LargeAccumulator`] the hot path doesn't touch.
#[derive(Clone, Debug)]
struct ColdState {
    /// Bitmap of chunk indices that are in use (in all lanes).
    used: [u64; USED_WORDS],
    /// Bitmap of non-zero words of `used`.
    used_summary: u64,
    /// Receives full chunks, infinities and NaNs.
    small: SmallAccumulator,
}

impl ColdState {
    const fn new() -> Self {
        Self {
            used: [0; USED_WORDS],
            used_summary: 0,
            small: SmallAccumulator::new(),
        }
    }

    /// Marks chunk `index` as used. Returns `false` if it already was.
    fn mark_used(&mut self, index: usize) -> bool {
        let bit = 1 << (index % 64);
        let newly_used = self.used[index / 64] & bit == 0;
        self.used[index / 64] |= bit;
        self.used_summary |= 1 << (index / 64);
        newly_used
    }

    fn used_indices(&self) -> impl Iterator<Item = usize> + '_ {
        used_indices(&self.used, self.used_summary)
    }
}

impl<const LANES: usize> Default for LargeAccumulator<LANES> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const LANES: usize> LargeAccumulator<LANES> {
    const LANES_IS_VALID: () = assert!(
        LANES > 0 && LANES <= 1 << 16,
        "between 1 and 2^16 lanes (so the combined lanes stay below 2^96)"
    );

    /// Creates an accumulator representing an empty sum (zero).
    #[must_use]
    pub fn new() -> Self {
        let () = Self::LANES_IS_VALID;
        Self {
            chunks: zeroed_table(),
            remaining: zeroed_table(),
            cold: ColdState::new(),
        }
    }

    /// Adds a single value.
    #[inline]
    pub fn add(&mut self, value: f64) {
        Self::add_in_lane(
            &mut self.chunks,
            &mut self.remaining,
            &mut self.cold,
            0,
            value.to_bits(),
        );
    }

    /// Adds all values of a slice.
    ///
    /// Accepts any element type that converts losslessly into `f64`, in
    /// particular `f32` and `f64`.
    pub fn add_slice<T: Copy + Into<f64>>(&mut self, values: &[T]) {
        let chunks = &mut *self.chunks;
        let remaining = &mut *self.remaining;
        let cold = &mut self.cold;
        let (blocks, remainder) = values.as_chunks::<LANES>();
        for block in blocks {
            for (lane, &value) in block.iter().enumerate() {
                Self::add_in_lane(chunks, remaining, cold, lane, value.into().to_bits());
            }
        }
        for (lane, &value) in remainder.iter().enumerate() {
            Self::add_in_lane(chunks, remaining, cold, lane, value.into().to_bits());
        }
    }

    /// Returns the exact sum held by this accumulator as a
    /// [`SmallAccumulator`], e.g. to merge partial sums.
    #[must_use]
    pub fn to_small(&self) -> SmallAccumulator {
        let mut small = self.cold.small.clone();
        for index in self.cold.used_indices() {
            // All lanes of an index share sign and exponent, so they are
            // combined into a single transfer.
            let magnitude = self.chunks[index]
                .iter()
                .zip(&self.remaining[index])
                .map(|(&sum, &remaining)| {
                    u128::from(chunk_magnitude(index, sum, MAX_COUNT - remaining))
                })
                .sum();
            transfer(&mut small, index, magnitude);
        }
        small
    }

    /// Resets the accumulator to zero while keeping its allocation.
    ///
    /// The cost is proportional to the number of chunks used, so reusing an
    /// accumulator is much cheaper than allocating a new one.
    pub fn clear(&mut self) {
        for index in self.cold.used_indices() {
            self.remaining[index] = [0; LANES];
        }
        self.cold = ColdState::new();
    }

    /// Returns the exact sum correctly rounded to the nearest `f64`.
    ///
    /// See [`SmallAccumulator::to_f64`] for the treatment of special values.
    #[must_use]
    pub fn to_f64(&self) -> f64 {
        self.to_small().to_f64()
    }

    /// Returns the exact sum correctly rounded to the nearest `f32`.
    ///
    /// See [`SmallAccumulator::to_f32`] for details.
    #[must_use]
    pub fn to_f32(&self) -> f32 {
        self.to_small().to_f32()
    }

    /// The hot path: adds a value (given by its bits) to a chunk of `lane`.
    #[inline]
    fn add_in_lane(
        chunks: &mut Table<u64, LANES>,
        remaining: &mut Table<u16, LANES>,
        cold: &mut ColdState,
        lane: usize,
        bits: u64,
    ) {
        let index = (bits >> MANTISSA_BITS) as usize;
        let (left, exhausted) = remaining[index][lane].overflowing_sub(1);
        remaining[index][lane] = left;
        if exhausted {
            Self::add_in_lane_slow(chunks, remaining, cold, lane, bits);
        } else {
            let chunk = &mut chunks[index][lane];
            *chunk = chunk.wrapping_add(bits);
        }
    }

    /// Handles specials, first use of a chunk index and full chunks. On entry,
    /// the hot path has wrapped the chunk's remaining capacity from 0 to
    /// `u16::MAX`.
    #[cold]
    #[inline(never)]
    fn add_in_lane_slow(
        chunks: &mut Table<u64, LANES>,
        remaining: &mut Table<u16, LANES>,
        cold: &mut ColdState,
        lane: usize,
        bits: u64,
    ) {
        let index = (bits >> MANTISSA_BITS) as usize;
        if is_special(bits) {
            // Special indices are never used, so they always take this path.
            remaining[index][lane] = 0;
            cold.small.add_special(bits);
            return;
        }
        if cold.mark_used(index) {
            chunks[index] = [0; LANES];
            remaining[index] = [MAX_COUNT; LANES];
        } else {
            // The index is in use, so this lane's chunk is full.
            let magnitude = chunk_magnitude(index, chunks[index][lane], MAX_COUNT);
            transfer(&mut cold.small, index, u128::from(magnitude));
        }
        chunks[index][lane] = bits;
        remaining[index][lane] = MAX_COUNT - 1;
    }
}

/// Iterates over the chunk indices marked in the `used` bitmap, whose non-zero
/// words are marked in `summary`.
fn used_indices(used: &[u64; USED_WORDS], summary: u64) -> impl Iterator<Item = usize> + '_ {
    set_bits(summary).flat_map(move |word| set_bits(used[word]).map(move |bit| word * 64 + bit))
}

/// Recovers the sum of the significands (implicit bits included) of the
/// `count` values whose raw bit patterns add up to `sum` in chunk `index`.
fn chunk_magnitude(index: usize, sum: u64, count: u16) -> u64 {
    let count = u64::from(count);
    let pattern_base = (index as u64) << MANTISSA_BITS;
    let mantissa_sum = sum.wrapping_sub(count.wrapping_mul(pattern_base));
    if index as u64 & EXP_MASK == 0 {
        mantissa_sum // subnormals and zeros have no implicit bit
    } else {
        mantissa_sum + count * IMPLICIT_BIT
    }
}

/// Adds `magnitude` significand units with the sign and exponent of chunk
/// `index` to `small`.
fn transfer(small: &mut SmallAccumulator, index: usize, magnitude: u128) {
    let biased_exp = (index as u64 & EXP_MASK).max(1);
    let negative = ((index as u64) << MANTISSA_BITS) & SIGN_BIT != 0;
    small.add_scaled(magnitude, biased_exp as u32, negative);
}

/// Iterates over the positions of the set bits of `word`, lowest first.
fn set_bits(mut word: u64) -> impl Iterator<Item = usize> {
    std::iter::from_fn(move || {
        (word != 0).then(|| {
            let bit = word.trailing_zeros() as usize;
            word &= word - 1;
            bit
        })
    })
}

/// Allocates a zero-initialised table on the heap (without a stack temporary).
fn zeroed_table<T: Copy + Default, const LANES: usize>() -> Box<Table<T, LANES>> {
    vec![[T::default(); LANES]; NUM_CHUNKS]
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("the vector has exactly NUM_CHUNKS elements"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum<const LANES: usize>(values: &[f64]) -> f64 {
        let mut acc = LargeAccumulator::<LANES>::new();
        acc.add_slice(values);
        acc.to_f64()
    }

    #[test]
    fn chunk_rollover_is_exact() {
        // Same sign and exponent: every value lands in the same chunk.
        let value = 1.999_999_999_999_999_8;
        for n in [2047, 2048, 2049, 4096, 4097, 3 * 2048 * 4 + 5] {
            let values = vec![value; n];
            let expected = value * n as f64;
            assert_eq!(sum::<1>(&values), expected, "n = {n}");
            assert_eq!(sum::<3>(&values), expected, "n = {n}");
            let negated: Vec<f64> = values.iter().map(|v| -v).collect();
            assert_eq!(sum::<4>(&negated), -expected, "n = {n}");
        }
    }

    #[test]
    fn lanes_of_an_index_fill_independently() {
        // Lane 0 fills the chunk of 1.5 and rolls over. The other lanes of the
        // same index hold two values of 1.25 each, and otherwise zeros.
        let values: Vec<f64> = (0..4 * (MAX_COUNT as usize + 1))
            .map(|i| match (i % 4, i < 8) {
                (0, _) => 1.5,
                (_, true) => 1.25,
                _ => 0.0,
            })
            .collect();
        let expected = f64::from(MAX_COUNT + 1) * 1.5 + 6.0 * 1.25;
        assert_eq!(sum::<4>(&values), expected);
    }

    #[test]
    fn subnormals_have_no_implicit_bit() {
        let tiny = f64::from_bits(0x000F_FFFF_FFFF_FFFF);
        let values = vec![tiny; 5000];
        assert_eq!(sum::<2>(&values), tiny * 5000.0);
    }

    #[test]
    fn tracks_used_chunks() {
        let mut acc = LargeAccumulator::<2>::new();
        acc.add_slice(&[1.0, -1.0, 3.0, 1e300, -1e-300, 0.0]);
        let used: Vec<usize> = acc.cold.used_indices().collect();
        let index = |v: f64| (v.to_bits() >> MANTISSA_BITS) as usize;
        let mut expected = vec![
            index(1.0),
            index(-1.0),
            index(3.0),
            index(1e300),
            index(-1e-300),
            index(0.0),
        ];
        expected.sort_unstable();
        assert_eq!(used, expected);
    }

    #[test]
    fn specials_bypass_chunks() {
        let mut acc = LargeAccumulator::<1>::new();
        acc.add_slice(&vec![f64::INFINITY; 5000]);
        assert_eq!(acc.cold.used_summary, 0);
        let index = (f64::INFINITY.to_bits() >> MANTISSA_BITS) as usize;
        assert_eq!(acc.remaining[index], [0]);
        assert_eq!(acc.to_f64(), f64::INFINITY);
        acc.add(f64::NEG_INFINITY);
        assert!(acc.to_f64().is_nan());
    }

    #[test]
    fn to_small_does_not_modify_state() {
        let mut acc = LargeAccumulator::<4>::new();
        acc.add_slice(&vec![0.5; 3000]);
        assert_eq!(acc.to_f64(), 1500.0);
        acc.add_slice(&[0.25; 4]);
        assert_eq!(acc.to_f64(), 1501.0);
    }

    #[test]
    fn clear_resets_to_zero() {
        let mut acc = LargeAccumulator::<2>::new();
        acc.add_slice(&[1.0, -3.5, 1e300, f64::NAN, 5e-324]);
        acc.add_slice(&vec![0.75; 5000]);
        acc.clear();
        assert_eq!(acc.to_f64().to_bits(), 0);
        assert_eq!(acc.cold.used_summary, 0);
        assert!(acc.remaining.iter().flatten().all(|&c| c == 0));
        acc.add_slice(&[2.0, 0.5, 1e300, -1e300]);
        assert_eq!(acc.to_f64(), 2.5);
    }

    #[test]
    fn set_bits_iterates_in_order() {
        let bits: Vec<usize> = set_bits(0b1001_0110 | 1 << 63).collect();
        assert_eq!(bits, [1, 2, 4, 7, 63]);
    }
}
