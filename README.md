# polars-xsum

**[Documentation](https://miraisolutions.github.io/polars-xsum/)** ·
[Changelog](https://miraisolutions.github.io/polars-xsum/changelog.html) ·
[Benchmarks](https://miraisolutions.github.io/polars-xsum/BENCHMARKS.html)

<!-- The docs/ pages include the sections between the "docs:" markers. Links use absolute URLs,
so that they also work on PyPI; the docs build turns links to the docs site into internal links. -->
<!-- docs:intro -->

**Exact, correctly rounded summation of floating-point columns for [Polars](https://pola.rs).**

`polars-xsum` is a Polars expression plugin that provides `xsum`, an exact replacement for
`pl.col(...).sum()` on float columns. It computes the **exact** sum of all values and rounds only
once at the end. It does this using an independent Rust implementation of
Radford M. Neal's *small and large superaccumulators*
([Neal 2015, arXiv:1505.05571](https://arxiv.org/abs/1505.05571)).
Long columns are summed at memory bandwidth on all cores.

```python
import polars as pl
from polars_xsum import xsum

df = pl.DataFrame({"x": [1e100, 1.0, -1e100]})
df.select(naive=pl.col("x").sum(), exact=xsum("x"))
# ┌───────┬───────┐
# │ naive ┆ exact │
# ╞═══════╪═══════╡
# │ 0.0   ┆ 1.0   │
# └───────┴───────┘
```

## Why?

Floating-point addition rounds after every step, so ordinary summation can lose any amount of
precision. It also depends on the order of the values and on how the data is chunked or
parallelised. `xsum` returns the exact sum rounded once to nearest (ties to even). The result is:

* **as accurate as possible**: the correctly rounded exact sum;
* **deterministic**: independent of value order, chunking and thread count;
* **robust**: intermediate overflow is impossible, so `[MAX, MAX, -MAX]` sums to `MAX`.

## Installation

```bash
pip install polars-xsum    # or: uv add polars-xsum
```

Supported versions: Python ≥ 3.11 and Polars ≥ 1.28. CI runs the full test suite against the
oldest supported Polars (1.28.0) and against the version pinned in `uv.lock`, which Dependabot
keeps current. Wheels are available for Linux with glibc (manylinux; x86_64 and aarch64),
Windows (x64) and macOS (x86_64 and arm64). On other platforms, such as Alpine Linux (musl),
`pip` builds the package from source, which requires Rust ≥ 1.95.

<!-- docs:guide -->

## Semantics

| Input | Result |
|---|---|
| `Float64` column | `Float64`, the exact sum correctly rounded |
| `Float32` column | `Float32`, rounded directly from the exact sum (no double rounding through `Float64`) |
| any other dtype (integers, `Float16`, `Decimal`, …) | `ComputeError`; cast explicitly to opt in |
| nulls | skipped; an empty or all-null column gives `0.0` (like `pl.sum`) |
| any NaN | NaN (for `Float64`, the payload of the first NaN is kept) |
| both `+inf` and `-inf` | NaN |
| otherwise an infinity | that infinity |
| exact sum beyond the finite range | `±inf` (incl. the tie at `MAX + ½ ulp`, as in IEEE-754) |
| exact zero | `+0.0` |

`xsum` is a scalar aggregation. It works in `select`, `with_columns` (broadcast),
`group_by(...).agg(...)`, `over(...)`, and in lazy and streaming queries. In `group_by` and
`over`, a single plugin call sums all groups. With overlapping windows (`rolling`,
`group_by_dynamic`), Polars gathers every window first, so memory grows with the number of rows
times the window length.

`xsum` also accepts a column name or a `Series`. To sum a `Series` directly, select the
expression:

```python
s = pl.Series([1e100, 1.0, -1e100])
pl.select(xsum(s)).item()  # 1.0
```

### Rolling sums

`rolling_xsum(expr, window_size, *, min_samples=None, center=False)` is the exact counterpart of
`pl.col(...).rolling_sum(...)`: every window of `window_size` rows is summed exactly and rounded
once, with the rules above. A single accumulator slides over the column, adding the value that
enters the window and subtracting the one that leaves it. Both steps are exact, so the cost per
row does not depend on the window size: ~1.7× the time of the inexact `rolling_sum`, more for
windows approaching the column length (see
[Benchmarks](https://miraisolutions.github.io/polars-xsum/BENCHMARKS.html)).

`rolling_xsum_by(expr, by, window_size, *, min_samples=0, closed="right")` is the exact
counterpart of `pl.col(...).rolling_sum_by(...)`: with the default `closed="right"`, the window of
a row holds the rows whose key lies in `(key - window_size, key]`. The key is an integer column
(`window_size="3i"`) or a `Date` or `Datetime` column, with a duration such as `"2d"`, `"1mo"` or
a `timedelta`; calendar units and time zones follow `dt.offset_by`. The key need not be sorted. This is about as fast as the
inexact `rolling_sum_by`, except for windows approaching the column length. The default
`min_samples=0` follows Polars 2 (Polars 1 defaults to 1).

For one sum per row, prefer these two to `xsum` in `rolling(...)`, which gathers every window
first. `xsum` remains the choice for `group_by_dynamic(...)`, which gives one row per window.

In `over(...)`, the windows of both functions stay within each group. Unlike `xsum`, which sums
all groups in one plugin call, they are called once per group, so with many small groups the
per-call overhead dominates: with 10^5 groups of 4 rows, they take about 2.5× the time of their
inexact counterparts.

## Threading

Columns of at least 32Ki values are summed in parallel. A plugin cannot use the thread pool of the
Polars host process: it is a separate library with its own copies of Polars and rayon. `xsum`
therefore has its own pool with threads named `xsum-0`, `xsum-1`, …, created on first use. It is
sized exactly like Polars' pool:

* `POLARS_MAX_THREADS` sets the number of threads, parsed with Polars' own configuration code;
* without it, the pool uses the number of available CPUs;
* `RAYON_NUM_THREADS` overrides the size for this plugin only.

`polars_xsum.thread_pool_size()` reports the size. While the pool works, the calling Polars
thread sleeps, so concurrent `xsum` expressions share the pool instead of oversubscribing the CPU.
The groups of a `group_by` or `over` are summed in parallel when they hold at least 32Ki values
in total. Shorter inputs are summed on the calling Polars thread. The result is bit-identical for
any number of threads.

## Not supported

* **Other dtypes**: integers, `Decimal` and `Float16` are rejected rather than converted
  silently. Cast them explicitly, keeping in mind that integers beyond 2^53 are not exactly
  representable as `Float64`.
* **Weighted rolling sums**: `rolling_xsum` has no `weights` parameter.
* **Other exact reductions**: there is no exact `sum_horizontal`, `cum_sum` or `mean`. A mean
  computed as `xsum("x") / pl.col("x").count()` is rounded twice (the sum, then the division), so
  it is within about one ulp of the exact mean, but not necessarily correctly rounded.
* **Method syntax**: `xsum` is a function, not an expression namespace, so write `xsum("x")` or
  `xsum(pl.col("x"))` rather than `pl.col("x").xsum()`.

<!-- docs:internals -->

## How it works

Every finite double is an integer multiple of `2^-1074`, the smallest subnormal, so a sum of
doubles is an integer times `2^-1074`. Neal's superaccumulators hold that integer exactly. They
count in units of `2^-1075` (a double with biased exponent `e` and 53-bit significand `m` is
`m · 2^(e − 1075)`, with `e` taken as 1 for subnormals), so that the exponent maps directly onto the digit positions.

* **Small superaccumulator**: 67 signed 64-bit "digits", each holding 32 bits of the
  fixed-point number plus 31 bits of headroom. A double with exponent `e` is split into two pieces
  added to digits `e/32` and `e/32 + 1`. Carries are propagated only once every 1023 additions.
* **Large superaccumulator**: one 64-bit chunk per sign and exponent (4096 chunks). A double is
  added by adding its *raw bit pattern* to the chunk selected by its top 12 bits. The mantissa
  sum can be recovered exactly as long as a chunk holds at most 2048 values. A full chunk is
  flushed into the small accumulator. A per-chunk counter of remaining capacity makes the common
  case a single decrement-and-branch plus one read-modify-write. Each chunk index has 4 *lanes*
  (separate chunks in one cache line) that consecutive values alternate between. This breaks the
  store-to-load dependency chains when many consecutive values share an exponent. The large
  accumulator handles about 2.3 G values/s on one core, ~20% faster than naive sequential `f64`
  summation on the same machine.
* **Rounding** extracts the top 53 bits of the exact integer plus guard and sticky bits. `Float32`
  results first round to `f64` with *round-to-odd*, after which the conversion to `f32` is
  provably correctly rounded.
* **Parallelism**: accumulators merge exactly, so long columns (≥ 32Ki values) are split into
  blocks that are summed on all cores with rayon. The result is bit-identical to the sequential
  sum. Columns and groups of fewer than 1,536 values use the small accumulator.
  Large accumulators are cleared and reused through a small per-thread cache (≤ 2 × 160 KiB per
  thread). This avoids paying allocation and page-fault costs on every call.
* **No `unsafe`**: `xsum-core` is `#![forbid(unsafe_code)]`. The plugin crate contains no
  hand-written unsafe code; its only unsafe code is the FFI glue that the `#[polars_expr]`
  (pyo3-polars) and `#[pymodule]`/`#[pyfunction]` (pyo3) macros generate.

See [Benchmarks](https://miraisolutions.github.io/polars-xsum/BENCHMARKS.html) for performance
measurements.

## References

Radford M. Neal, *Fast exact summation using small and large superaccumulators*, 2015.
[arXiv:1505.05571](https://arxiv.org/abs/1505.05571)

<!-- docs:end -->

## Contributing

See [CONTRIBUTING.md](https://github.com/miraisolutions/polars-xsum/blob/main/CONTRIBUTING.md)
for the development setup, tests, quality gates, continuous integration and the release process.

## License

[MIT](https://github.com/miraisolutions/polars-xsum/blob/main/LICENSE)
