# polars-xsum

**[Documentation](https://miraisolutions.github.io/polars-xsum/)** ·
[Changelog](CHANGELOG.md) ·
[Benchmarks](BENCHMARKS.md)

<!-- The docs/ pages include the sections between the "docs:" markers. -->
<!-- docs:intro -->

**Exact, correctly rounded summation of floating-point columns for [Polars](https://pola.rs).**

`polars-xsum` is a Polars expression plugin that provides `xsum`, a drop-in replacement for
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

Supported versions: Python ≥ 3.11 and Polars ≥ 1.28 (the full test suite passes from Polars 1.28.0
to the latest release; CI checks both ends). Wheels are available for Linux (x86_64 and aarch64),
Windows (x64) and macOS (x86_64 and arm64). On other platforms, `pip` builds the package from
source, which requires Rust ≥ 1.95.

<!-- docs:guide -->

## Semantics

| Input | Result |
|---|---|
| `Float64` column | `Float64`, the exact sum correctly rounded |
| `Float32` column | `Float32`, rounded directly from the exact sum (no double rounding through `Float64`) |
| any other dtype (integers, `Float16`, `Decimal`, …) | `ComputeError`; cast explicitly to opt in |
| nulls | skipped; an empty or all-null column gives `0.0` (like `pl.sum`) |
| any NaN | NaN (the payload of the first NaN is kept) |
| `+inf` and `-inf` | NaN |
| otherwise an infinity | that infinity |
| exact sum beyond the finite range | `±inf` (incl. the tie at `MAX + ½ ulp`, as in IEEE-754) |
| exact zero | `+0.0` |

`xsum` is a scalar aggregation. It works in `select`, `with_columns` (broadcast),
`group_by(...).agg(...)`, `over(...)`, and in lazy and streaming queries. In `group_by` and
`over`, a single plugin call sums all groups. With overlapping windows (`rolling`,
`group_by_dynamic`), Polars gathers every window first, so memory grows with the number of rows
times the window length.

### Rolling sums

`rolling_xsum(expr, window_size, *, min_samples=None, center=False)` is the exact counterpart of
`pl.col(...).rolling_sum(...)`: every window of `window_size` rows is summed exactly and rounded
once, with the rules above. A single accumulator slides over the column, adding the value that
enters the window and subtracting the one that leaves it. Both steps are exact, so the cost per
row does not depend on the window size: ~1.7× the time of the inexact `rolling_sum`, more for
windows approaching the column length (see [BENCHMARKS.md](BENCHMARKS.md)).

`rolling_xsum_by(expr, by, window_size, *, min_samples=0, closed="right")` is the exact
counterpart of `pl.col(...).rolling_sum_by(...)`: the window of a row holds the rows whose key lies
in `(key - window_size, key]`. The key is an integer column (`window_size="3i"`) or a `Date` or
`Datetime` column, with a duration such as `"2d"`, `"1mo"` or a `timedelta`; calendar units and
time zones follow `dt.offset_by`. The key need not be sorted. This is about as fast as the
inexact `rolling_sum_by`, except for windows approaching the column length. The default
`min_samples=0` follows Polars 2 (Polars 1 defaults to 1).

For one sum per row, prefer these two to `xsum` in `rolling(...)`, which gathers every window
first. `xsum` remains the choice for `group_by_dynamic(...)`, which gives one row per window.

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

<!-- docs:development -->

## Development

Development requires Rust ≥ 1.95 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # creates .venv and builds the extension (release mode)
uv run pytest              # Python test suite (incl. doctests and Hypothesis properties)
cargo test --workspace     # Rust unit, integration, property and doc tests
cargo bench -p xsum-core   # criterion benchmarks (see BENCHMARKS.md)
uv run python benchmarks/bench_polars.py
uv run --group docs sphinx-build -W --keep-going -n docs docs/_build/html   # documentation
```

The documentation (`docs/`) is built with Sphinx. Most of its text is included from this README,
`BENCHMARKS.md` and `CHANGELOG.md`; the API reference comes from the docstrings. Open
`docs/_build/html/index.html` to view it.

After changing Rust code, `uv sync --reinstall-package polars-xsum` rebuilds the extension.
`uv run` also rebuilds it automatically when sources change. `uv run maturin develop --release`
works too.

Quality gates:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings   # clippy::pedantic enabled
uv run ruff format --check . && uv run ruff check .     # ruff with all rules
uv run ty check                                         # Astral's type checker
```

## Continuous integration and releases

`.github/workflows/ci.yml` runs on every push and pull request:

* **Rust:** formatting, clippy (`-D warnings`), tests and docs, plus a build with the minimum
  supported Rust version (1.95).
* **Python:** ruff and ty, and the test suite on Linux, macOS and Windows with Python 3.11 and
  3.14, plus a run against the oldest supported Polars (1.28.0).
* **Wheels:** abi3 wheels for Linux (x86_64 and aarch64, manylinux), Windows x64 and macOS
  (x86_64 and arm64), plus an sdist. The test suite then runs against the *installed* wheels on
  each native platform (including Intel macOS), and against a build from the sdist.
* **Documentation:** a Sphinx build in which any warning, including a broken cross-reference, is an
  error.

Actions are pinned to commit SHAs and kept current by Dependabot (`.github/dependabot.yml`). The
workflow passes [actionlint](https://github.com/rhysd/actionlint) and
[zizmor](https://docs.zizmor.sh).

**Releasing:**

1. Bump `version` in `Cargo.toml`, and rename the `Unreleased` section of `CHANGELOG.md` to
   `[X.Y.Z] - YYYY-MM-DD`.
2. Push a tag `vX.Y.Z` with the same version.

The workflow checks that the tag matches the version and that `CHANGELOG.md` has a dated section
for it. It builds everything from scratch (no caches) and publishes to PyPI with trusted
publishing and build-provenance attestations. It then creates the GitHub release, with that
changelog section as its notes and the wheels and the sdist attached, and deploys the
documentation to GitHub Pages. The published documentation therefore always describes the latest
release.

One-time setup:

1. On PyPI, add a trusted publisher for the project `polars-xsum`: the GitHub owner and
   repository, workflow `ci.yml`, environment `pypi`.
2. In the GitHub repository settings, create the environment `pypi` (optionally with required
   reviewers).
3. In the GitHub repository settings, under *Pages*, set the source to *GitHub Actions*. This
   creates the environment `github-pages`; allow deployments from tags `v*` in its deployment
   rules (by default only the default branch may deploy).

## Project layout

```
crates/xsum-core/     pure Rust library: the superaccumulators (no Polars dependency)
  src/ieee.rs           IEEE-754 binary64 layout helpers
  src/small.rs          small superaccumulator (67 × 32-bit digits in i64)
  src/large.rs          large superaccumulator (4096 sign/exponent chunks, lanes)
  src/round.rs          exact → f64 rounding (nearest-even, round-to-odd for f32)
  src/parallel.rs       rayon fold/reduce over blocks
  tests/                edge cases + proptest against a num-bigint oracle
  benches/sum.rs        criterion benchmarks
crates/polars-xsum/   the Polars plugin (cdylib): chunk/null handling, dtype checks
python/polars_xsum/   Python API: `xsum(expr)`
tests/                pytest suite with an exact `fractions.Fraction` oracle
benchmarks/           Python-level benchmarks
docs/                 Sphinx documentation (includes parts of README.md)
.github/              CI workflow, Dependabot, release-notes script
```

<!-- docs:internals -->

## How it works

Every finite double is an integer multiple of `2^-1075`, so a sum of doubles is an integer times
`2^-1075`. Neal's superaccumulators hold that integer exactly.

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
  sum. Short columns (e.g. `group_by` groups) use the small accumulator.
  Large accumulators are cleared and reused through a small per-thread cache (≤ 2 × 160 KiB per
  thread). This avoids paying allocation and page-fault costs on every call.
* **No `unsafe`**: `xsum-core` is `#![forbid(unsafe_code)]`. In the plugin crate, the only unsafe
  code is the FFI entry points that the `#[polars_expr]` macro generates.

See [BENCHMARKS.md](BENCHMARKS.md) for performance measurements.

## References

Radford M. Neal, *Fast exact summation using small and large superaccumulators*, 2015.
[arXiv:1505.05571](https://arxiv.org/abs/1505.05571)

<!-- docs:end -->

## License

[MIT](LICENSE)
