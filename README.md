# polars-xsum

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
`group_by(...).agg(...)`, `over(...)`, and in lazy and streaming queries.

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
Shorter columns, such as typical `group_by` groups, are summed on the calling Polars thread. The
result is bit-identical for any number of threads.

## Installation and development

Supported versions: Python ≥ 3.11 and Polars ≥ 1.0 (the full test suite passes from Polars 1.0.0
to the latest release; CI checks both ends). Building from source requires Rust ≥ 1.95 and
[uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # creates .venv and builds the extension (release mode)
uv run pytest              # Python test suite (incl. doctests and Hypothesis properties)
cargo test --workspace     # Rust unit, integration, property and doc tests
cargo bench -p xsum-core   # criterion benchmarks (see BENCHMARKS.md)
uv run python benchmarks/bench_polars.py
```

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
  3.14, plus a run against the oldest supported Polars (1.0.0).
* **Wheels:** abi3 wheels for Linux (x86_64 and aarch64, manylinux), Windows x64 and macOS
  (x86_64 and arm64), plus an sdist. The test suite then runs against the *installed* wheels on
  each native platform (including Intel macOS), and against a build from the sdist.

Actions are pinned to commit SHAs and kept current by Dependabot (`.github/dependabot.yml`). The
workflow passes [actionlint](https://github.com/rhysd/actionlint) and
[zizmor](https://docs.zizmor.sh).

**Releasing:** bump `version` in `Cargo.toml`, then push a tag `vX.Y.Z` with the same version. The
workflow checks that the tag matches the version, builds everything from scratch (no caches) and
publishes to PyPI with trusted publishing and build-provenance attestations.

One-time setup:

1. On PyPI, add a trusted publisher for the project `polars-xsum`: the GitHub owner and
   repository, workflow `ci.yml`, environment `pypi`.
2. In the GitHub repository settings, create the environment `pypi` (optionally with required
   reviewers).

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
```

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
  sum. Short columns (e.g. `group_by` groups) use the small accumulator on the calling thread.
  Large accumulators are cleared and reused through a small per-thread cache (≤ 2 × 160 KiB per
  thread). This avoids paying allocation and page-fault costs on every call.
* **No `unsafe`**: `xsum-core` is `#![forbid(unsafe_code)]`. In the plugin crate, the only unsafe
  code is the FFI entry points that the `#[polars_expr]` macro generates.

See [BENCHMARKS.md](BENCHMARKS.md) for performance measurements.

## References

Radford M. Neal, *Fast exact summation using small and large superaccumulators*, 2015.
[arXiv:1505.05571](https://arxiv.org/abs/1505.05571)

## License

[MIT](LICENSE)
