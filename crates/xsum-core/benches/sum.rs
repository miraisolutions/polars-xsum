//! Criterion benchmarks for exact summation.
//!
//! Run with `cargo bench -p xsum-core`. Individual groups can be selected with
//! a filter, e.g. `cargo bench -p xsum-core -- algorithms`.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use xsum_core::{LargeAccumulator, SmallAccumulator};

/// Seed of the random number generator, so every run uses the same inputs.
const SEED: u64 = 0x5EED;

/// Benchmark input distributions.
#[derive(Clone, Copy, Debug)]
enum Data {
    /// Uniform in `[0, 1)`: few distinct exponents, all positive.
    Uniform,
    /// Normal with mean 0 and standard deviation 1000: both signs, ~30 exponents.
    Normal,
    /// Random finite bit patterns: exponents spread over the whole range.
    Wide,
    /// Pairs `x, -x` (shuffled) plus small noise: massive cancellation.
    Cancelling,
}

impl Data {
    /// All distributions, in the order they are reported.
    const ALL: [Self; 4] = [Self::Uniform, Self::Normal, Self::Wide, Self::Cancelling];

    /// Returns `n` values drawn from this distribution.
    fn generate(self, n: usize) -> Vec<f64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(SEED);
        match self {
            Self::Uniform => (0..n).map(|_| rng.random::<f64>()).collect(),
            Self::Normal => {
                let normal = Normal::new(0.0, 1000.0).expect("valid parameters");
                (0..n).map(|_| normal.sample(&mut rng)).collect()
            }
            Self::Wide => (0..n)
                .map(|_| {
                    loop {
                        let value = f64::from_bits(rng.next_u64());
                        if value.is_finite() {
                            break value;
                        }
                    }
                })
                .collect(),
            Self::Cancelling => {
                let mut values: Vec<f64> = (0..n / 2)
                    .flat_map(|_| {
                        let x = rng.random_range(-1e10..1e10);
                        [x, -x + rng.random_range(-1e-6..1e-6)]
                    })
                    .collect();
                values.resize(n, 1.0);
                // Fisher-Yates shuffle.
                for i in (1..values.len()).rev() {
                    values.swap(i, rng.random_range(0..=i));
                }
                values
            }
        }
    }
}

/// Plain (inexact) floating-point summation as a speed reference.
fn naive_sum(values: &[f64]) -> f64 {
    values.iter().sum()
}

/// Neumaier's improved Kahan summation (more accurate, still inexact).
fn neumaier_sum(values: &[f64]) -> f64 {
    let (mut sum, mut compensation) = (0.0_f64, 0.0_f64);
    for &v in values {
        let t = sum + v;
        compensation += if sum.abs() >= v.abs() {
            (sum - t) + v
        } else {
            (v - t) + sum
        };
        sum = t;
    }
    sum + compensation
}

/// Exact sum with a [`SmallAccumulator`].
fn small_sum(values: &[f64]) -> f64 {
    let mut acc = SmallAccumulator::new();
    acc.add_slice(values);
    acc.to_f64()
}

/// Exact sum with a [`LargeAccumulator`] with `LANES` lanes.
fn large_sum<const LANES: usize>(values: &[f64]) -> f64 {
    let mut acc = LargeAccumulator::<LANES>::new();
    acc.add_slice(values);
    acc.to_f64()
}

/// Sizes from 10^2 to 10^7 (10^8 with `XSUM_BENCH_HUGE=1`).
fn sizes() -> Vec<usize> {
    let max_exp = if std::env::var_os("XSUM_BENCH_HUGE").is_some() {
        8
    } else {
        7
    };
    (2..=max_exp).map(|e| 10_usize.pow(e)).collect()
}

/// Sets the throughput unit to `n` elements and adapts the sampling to the
/// input size, so that large inputs do not take too long.
fn configure(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    n: usize,
) {
    group.throughput(Throughput::Elements(n as u64));
    if n >= 1_000_000 {
        group
            .sample_size(20)
            .measurement_time(Duration::from_secs(4));
    } else {
        group
            .sample_size(50)
            .measurement_time(Duration::from_secs(2));
    }
    group.warm_up_time(Duration::from_millis(500));
}

/// All algorithms over all sizes on uniform data.
fn algorithms(c: &mut Criterion) {
    type Algorithm = (&'static str, fn(&[f64]) -> f64);
    let algorithms: [Algorithm; 10] = [
        ("naive", naive_sum),
        ("neumaier", neumaier_sum),
        ("small", small_sum),
        ("large_lanes1", large_sum::<1>),
        ("large_lanes2", large_sum::<2>),
        ("large_lanes4", large_sum::<4>),
        ("large_lanes8", large_sum::<8>),
        ("sum_f64", xsum_core::sum_f64),
        ("par_sum_f64", xsum_core::par_sum_f64),
        ("par_naive", |v| v.par_iter().sum()),
    ];
    let mut group = c.benchmark_group("algorithms");
    for n in sizes() {
        let values = Data::Uniform.generate(n);
        configure(&mut group, n);
        for (name, algorithm) in algorithms {
            group.bench_with_input(BenchmarkId::new(name, n), &values, |b, v| {
                b.iter(|| algorithm(black_box(v)));
            });
        }
    }
    group.finish();
}

/// Sensitivity to the input distribution at a fixed size.
fn distributions(c: &mut Criterion) {
    type Algorithm = (&'static str, fn(&[f64]) -> f64);
    let algorithms: [Algorithm; 5] = [
        ("naive", naive_sum),
        ("small", small_sum),
        ("large_lanes4", large_sum::<4>),
        ("sum_f64", xsum_core::sum_f64),
        ("par_sum_f64", xsum_core::par_sum_f64),
    ];
    let n = 1_000_000;
    let mut group = c.benchmark_group("distributions");
    configure(&mut group, n);
    for data in Data::ALL {
        let values = data.generate(n);
        for (name, algorithm) in algorithms {
            group.bench_with_input(
                BenchmarkId::new(name, format!("{data:?}")),
                &values,
                |b, v| {
                    b.iter(|| algorithm(black_box(v)));
                },
            );
        }
    }
    group.finish();
}

/// `f32` input (widened exactly, rounded via round-to-odd).
fn single_precision(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32");
    for n in [1_000, 1_000_000] {
        let values: Vec<f32> = Data::Normal.generate(n).iter().map(|&v| v as f32).collect();
        configure(&mut group, n);
        group.bench_with_input(BenchmarkId::new("naive", n), &values, |b, v| {
            b.iter(|| black_box(v).iter().sum::<f32>());
        });
        group.bench_with_input(BenchmarkId::new("sum_f32", n), &values, |b, v| {
            b.iter(|| xsum_core::sum_f32(black_box(v)));
        });
        group.bench_with_input(BenchmarkId::new("par_sum_f32", n), &values, |b, v| {
            b.iter(|| xsum_core::par_sum_f32(black_box(v)));
        });
    }
    group.finish();
}

/// Strong scaling of the parallel sum with the number of threads.
fn thread_scaling(c: &mut Criterion) {
    let n = 10_000_000;
    let values = Data::Normal.generate(n);
    let max_threads = std::thread::available_parallelism().map_or(1, usize::from);
    let mut group = c.benchmark_group("threads");
    configure(&mut group, n);
    for threads in [1, 2, 4, 8, 12, 16, 24]
        .into_iter()
        .filter(|&t| t <= max_threads)
    {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        group.bench_with_input(BenchmarkId::new("par_sum_f64", threads), &values, |b, v| {
            b.iter(|| pool.install(|| xsum_core::par_sum_f64(black_box(v))));
        });
    }
    group.finish();
}

/// Work-item size of the parallel sum.
fn block_size(c: &mut Criterion) {
    let n = 10_000_000;
    let values = Data::Normal.generate(n);
    let mut group = c.benchmark_group("block_size");
    configure(&mut group, n);
    for block in [1 << 12, 1 << 14, 1 << 15, 1 << 16, 1 << 18, 1 << 20] {
        group.bench_with_input(
            BenchmarkId::new("par_sum_blocks", block),
            &values,
            |b, v| {
                b.iter(|| {
                    xsum_core::par_sum_blocks(black_box(v).par_chunks(block), |acc, chunk| {
                        acc.add_slice(chunk);
                    })
                    .to_f64()
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    algorithms,
    distributions,
    single_precision,
    thread_scaling,
    block_size
);
criterion_main!(benches);
