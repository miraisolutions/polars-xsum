# Benchmarks

This document reports the performance of the Rust core (`xsum-core`, criterion benchmarks) and of
the Polars plugin (`polars_xsum.xsum`, Python-level timings).

## Summary

* **Single core:** the large superaccumulator sums **~2.3 G values/s** (0.44 ns/value). That
  is ~20% faster than naive sequential `f64` summation (`iter().sum()`, latency-bound at ~1.9 G/s)
  and ~2× faster than Neumaier compensated summation, which is still inexact.
* **All cores:** long inputs reach **~15 G values/s** (~120 GB/s) while the data is in cache. From
  10^7 values on, the sum runs at memory bandwidth (~12 G values/s) and is as fast as rayon's
  parallel naive sum.
* **Through Polars:** for columns of 10^7 values or more, `xsum` costs the same as Polars' own
  inexact `sum` (≈ 65 GB/s).
* **Exactness costs essentially nothing at scale.** The cost is visible for short columns (a few
  µs of extra fixed overhead) and in `group_by` with few groups, where Polars' generic plugin
  aggregation path dominates (see below).

## Setup

| | |
|---|---|
| CPU | AMD Ryzen 9 7900X, 12 cores / 24 threads, AVX2 + AVX-512 (L2 1 MiB/core, L3 2 × 32 MiB) |
| Memory | 32 GB |
| OS | Linux 7.2.3 (Manjaro) |
| Rust | 1.99.1 stable, `--release` with `lto = "fat"`, `codegen-units = 1`, default target (x86-64-v1) |
| Python | 3.13.18, polars 2.1.0, numpy 2.5.4 |

Numbers are criterion's point estimates (median-based) or, for Python, the median of 7 runs after a
warm-up. Run-to-run variation is typically below ±5%. Timings through Polars below 10^6 values are
dominated by fixed per-query overheads and thread wake-up, so they vary more.

### Reproduce

```bash
cargo bench -p xsum-core                         # all groups, ~10 min
cargo bench -p xsum-core -- algorithms           # a single group
XSUM_BENCH_HUGE=1 cargo bench -p xsum-core       # include 10^8 values
uv run python benchmarks/bench_polars.py         # Polars-level tables
```

## Rust core (`cargo bench -p xsum-core`)

Input: uniform `[0, 1)` unless noted. Throughput in values per second (higher is better).
`naive` = `iter().sum()`, `neumaier` = compensated summation (both inexact),
`small` / `large_lanesL` = the accumulators used directly, `sum_f64` = automatic choice,
`par_sum_f64` = multi-threaded (24 threads), `par_naive` = `par_iter().sum()` (inexact).

| n | naive | neumaier | small | large ×1 | large ×2 | large ×4 | large ×8 | `sum_f64` | `par_sum_f64` | par_naive |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10^2 | 3.88 G | 1.17 G | **554 M** | 235 M | 153 M | 87 M | 47 M | 480 M | 472 M | 6.3 M |
| 10^3 | 1.90 G | 1.17 G | 746 M | **1.01 G** | 861 M | 609 M | 385 M | 691 M | 682 M | 37 M |
| 10^4 | 1.84 G | 1.17 G | 774 M | 1.53 G | 1.77 G | 1.85 G | 1.51 G | **2.22 G** | 2.24 G | 264 M |
| 10^5 | 1.88 G | 1.14 G | 770 M | 1.55 G | 1.98 G | 2.18 G | 1.99 G | 2.27 G | **5.30 G** | 1.79 G |
| 10^6 | 1.87 G | 1.17 G | 774 M | 1.60 G | 2.05 G | 2.26 G | 2.16 G | 2.27 G | **15.1 G** | 10.2 G |
| 10^7 | 1.87 G | 1.16 G | 774 M | 1.59 G | 2.04 G | 2.24 G | 2.13 G | 2.24 G | **12.0 G** | 11.6 G |

Observations:

* **Small vs large accumulator.** The small accumulator has a tiny state (0.5 KiB) and wins for
  short inputs. The large one holds ~160 KiB of zero-initialised state per accumulator, which is
  amortised from a few thousand values on. `sum_f64` switches at 1,536 values. It takes its large
  accumulators from a per-thread cache, which makes it faster than a freshly allocated `large ×4`
  at 10^4 values (2.22 G vs 1.85 G).
* **Hot path.** Adding a value to the large accumulator takes about 8 instructions: a
  decrement-and-branch on the chunk's remaining capacity and a read-modify-write of the chunk. The
  loop is bound by op throughput and memory operations, not by latency. Counting up and testing
  the range of the count, as before, took ~14 instructions per value (1.82 G values/s). A
  count-free variant that accumulates significands and flushes all chunks every `2048 · LANES`
  values was at most ~10% faster on typical data. It was 3–4× slower on data with many zeros or
  with spread-out exponents, so it was not adopted.
* **Lanes.** With one lane, consecutive values with the same exponent update the same memory word,
  forming a load→add→store dependency chain through the store buffer. Four lanes (separate chunks
  in the same cache line, used round-robin) break the chain: 1.60 G → 2.26 G values/s. Eight lanes
  are slower again (larger state, more chunks to flush). **4 lanes is the default.**
* **Parallel scaling** is limited by memory bandwidth for large n, not by the algorithm. While the
  data is in cache (10^6 values), the exact `par_sum_f64` is 1.5× faster than the inexact parallel
  naive sum, which is latency-bound per core. At 10^7 values (80 MB) both run at memory
  bandwidth.

### Input distribution (n = 10^6)

| distribution | naive | small | large ×4 | `sum_f64` | `par_sum_f64` |
|---|---:|---:|---:|---:|---:|
| uniform `[0, 1)` | 1.88 G | 776 M | 2.25 G | 2.30 G | 15.2 G |
| normal, σ = 1000 | 1.87 G | 763 M | 2.44 G | 2.46 G | 15.3 G |
| cancelling `x, −x` pairs | 1.87 G | 733 M | 2.40 G | 2.43 G | 14.9 G |
| random bit patterns ("wide") | 1.86 G | 765 M | 1.54 G | 1.54 G | 3.87 G |

Throughput does not depend on the values (signs, cancellation, magnitudes) as long as they span a
realistic range of exponents. The adversarial "wide" input draws exponents uniformly from the whole
`2^±1023` range. That spreads 10^6 values over all 4096 sign/exponent chunks × 4 lanes, so most
additions take the slow path that initialises a chunk. In parallel, every task has to initialise
its own chunks and flush ~16K chunks at the end, which limits scaling. Two measures recovered most
of that cost:

* lanes are combined into a single 128-bit transfer per chunk index;
* each rayon task processes a run of consecutive blocks (≈ 4 tasks per thread), so fewer
  accumulators are needed. This took 10^7 wide values from 3.7 ms to 1.2 ms.

### `f32` input

| n | naive `f32` | `sum_f32` | `par_sum_f32` |
|---:|---:|---:|---:|
| 10^3 | 1.97 G | 584 M | 581 M |
| 10^6 | 1.88 G | 1.69 G | 10.7 G |

`f32` values are widened exactly to `f64` on the fly (one conversion instruction per value), and the
result is rounded to `f32` via round-to-odd. This runs at ~70–75% of the `f64` throughput, both
single-threaded and in parallel.

### Thread scaling (n = 10^7, normal data)

| threads | 1 | 2 | 4 | 8 | 12 | 16 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|
| values/s | 2.21 G | 4.33 G | 8.49 G | 12.2 G | 12.8 G | 11.4 G | 12.0 G |
| speed-up | 1.0× | 2.0× | 3.8× | 5.5× | 5.8× | 5.2× | 5.4× |

Scaling is near-linear up to 4 threads. It saturates at ~12 G values/s ≈ 100 GB/s. That is
mostly served from the 64 MiB L3 (the 80 MB input partly fits), and beyond that from DRAM. SMT
threads bring no further gain. The algorithm is load/store-port bound, and both hyperthreads of a
core share those ports.

### Block size of the parallel sum (n = 10^7)

| block | 4Ki | 16Ki | 32Ki | 64Ki | 256Ki | 1Mi |
|---|---:|---:|---:|---:|---:|---:|
| values/s | 11.7 G | 11.7 G | 11.8 G | 12.0 G | 11.8 G | 10.6 G |

Blocks between 4Ki and 256Ki values perform the same. Only blocks too coarse for 24 threads lose
throughput. The library uses `len / (4 · threads)` clamped to `[4Ki, 16Ki]`. Inputs below 32Ki
values are summed serially (parallelism did not pay off below that in a separate sweep).

### SIMD and `target-cpu=native`

Both hot loops are scatter-add operations: every value updates a chunk selected by its own
exponent, with frequent conflicts between neighbouring values. Neither LLVM's auto-vectoriser
nor AVX-512 can usefully vectorise that without conflict-detection intrinsics, and those would
require `unsafe`. Compiling with `RUSTFLAGS="-C target-cpu=native"` (AVX-512 enabled):

| benchmark | default target | `target-cpu=native` |
|---|---:|---:|
| large ×4, 10^6 | 2.26 G | 2.29 G |
| small, 10^6 | 774 M | 552 M |
| small, 10^3 | 746 M | 509 M |

AVX-512 brings no benefit, and the small accumulator even gets slower: LLVM's AVX-512 code for the
digit split costs more than it saves. Runtime CPU-feature dispatch (e.g. via `multiversion`) was
therefore not adopted. The data-parallel resources of modern CPUs are exploited instead through:

* **instruction-level parallelism**: independent lanes in the large accumulator;
* **multiple cores**: rayon with exact merging;
* **cache efficiency**: only the chunks for occurring exponents are touched, and accumulators are
  reused per thread.

The helper loops around the core, such as `f32 → f64` widening and the masked iteration over
validity bitmaps, are written so the compiler can vectorise them. Nevertheless, scattering into the
accumulator dominates.

## Polars plugin (`uv run python benchmarks/bench_polars.py`)

### Accuracy

Input: 10^6 values, formed as ±N(0, 10¹²) pairs plus 10,000 N(0, 1) values, shuffled. The exact sum
is computed with `fractions.Fraction`.

| method | result | relative error |
|---|---:|---:|
| `pl.col("x").sum()` | `-119.9375` | 9.8e-04 |
| `np.sum` (pairwise) | `-120.125` | 2.6e-03 |
| `math.fsum` | `-119.8196730748211` | 0 |
| `xsum("x")` | `-119.8196730748211` | 0 |

`math.fsum` is also correctly rounded, but it is 16× (10^4 values) to 280× (10^7 values) slower
than `xsum` (see below). It also
raises `OverflowError` on intermediate overflow, which `xsum` handles exactly.

### Whole-column sum, Float64 (normal data)

| n | `pl.sum` | `xsum` | `np.sum` | `math.fsum` | `xsum` ns/value | `xsum` GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 10^3 | 9.6 µs | 14.8 µs | 1.1 µs | 27.5 µs | 14.8 | 0.5 |
| 10^4 | 8.4 µs | 16.6 µs | 2.1 µs | 265 µs | 1.66 | 4.8 |
| 10^5 | 149 µs | 125 µs | 13.3 µs | 3.02 ms | 1.25 | 6.4 |
| 10^6 | 142 µs | 154 µs | 130 µs | 27.5 ms | 0.15 | 52.0 |
| 10^7 | 0.95 ms | 1.00 ms | 1.47 ms | 282 ms | 0.10 | 79.8 |
| 10^8 | 12.1 ms | 11.9 ms | 14.9 ms | – | 0.12 | 67.0 |

* For small columns, the time is dominated by Polars' per-query overhead (~9 µs for `pl.sum`).
  The plugin adds a few µs of FFI overhead on top.
* Between 10^5 and 10^6 values, both Polars and the plugin spend much of their time waking their
  thread pools, so those rows vary noticeably from run to run.
* From 10^7 values on, `xsum` runs at memory bandwidth, within a few percent of Polars' inexact
  sum and faster than NumPy.

### Float32, nulls

| case | `pl.sum` | `xsum` |
|---|---:|---:|
| Float32, 10^4 | 8.4 µs | 18.0 µs |
| Float32, 10^6 | 147 µs | 182 µs |
| Float32, 10^7 | 471 µs | 658 µs |
| Float64 with 10% nulls, 10^7 | 0.996 ms | 1.01 ms |

`Float32` columns have half the bytes per value, so Polars' SIMD sum gets twice the throughput in
values/s. `xsum` is compute-bound here at ~15 G values/s. Nulls are handled 64 values at a time
from the validity bitmap and cost almost nothing.

### `group_by(...).agg(...)`, n = 10^7

| groups | `pl.sum` | `xsum` | ratio |
|---:|---:|---:|---:|
| 10 | 4.2 ms | 66.4 ms | 15.8× |
| 1,000 | 26 ms | 58 ms | 2.2× |
| 100,000 | 30 ms | 141 ms | 4.7× |

This overhead is **not** the summation. Polars computes its built-in `sum` directly on the group
indices. For any opaque aggregation, plugin or Python UDF alike, Polars first gathers every group
into its own Series and then calls the function once per group. Replacing `xsum` with
`map_batches(lambda s: s.sum(), returns_scalar=True)` costs the same ~66 ms for 10 groups. With
many small groups the per-call cost of the plugin interface (~1 µs per group) adds up. Running the
plugin single-threaded (`RAYON_NUM_THREADS=1`) does not change these numbers, so there is no
thread oversubscription.
