# Contributing

<!-- docs/development.md includes everything below this marker. -->
<!-- docs:start -->

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

The documentation (`docs/`) is built with Sphinx. Most of its text is included from `README.md`,
`CONTRIBUTING.md`, `BENCHMARKS.md` and `CHANGELOG.md`; the API reference comes from the docstrings.
Open `docs/_build/html/index.html` to view it.

`README.md` is also the project description on PyPI, where relative links do not work. Its links
therefore use absolute URLs. The documentation build turns links to the documentation site
(`https://miraisolutions.github.io/polars-xsum/...`) back into links between its pages, so that
they are checked by the build.

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

* **Rust:** formatting, clippy (`-D warnings`), tests and docs, plus `cargo check` with the
  minimum supported Rust version (1.95).
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

1. Bump `version` in `Cargo.toml` and run `cargo update --workspace`, which records the new
   version in `Cargo.lock` (CI runs cargo with `--locked`, so a stale lock file fails the
   release). Rename the `Unreleased` section of `CHANGELOG.md` to `[X.Y.Z] - YYYY-MM-DD`, and at
   its bottom replace the `[Unreleased]` link definition with
   `[X.Y.Z]: https://github.com/miraisolutions/polars-xsum/releases/tag/vX.Y.Z`. When a new
   `Unreleased` section is started later, define it as
   `[Unreleased]: https://github.com/miraisolutions/polars-xsum/compare/vX.Y.Z...HEAD`.
   Commit these changes to `main`.
2. Push a tag `vX.Y.Z` with the same version.

The workflow checks that the tag matches the version and that `CHANGELOG.md` has a dated section
for it. It builds everything from scratch (no caches) and publishes to PyPI with trusted
publishing and build-provenance attestations. It then creates the GitHub release, with that
changelog section as its notes and the wheels and the sdist attached, and deploys the
documentation to GitHub Pages. The published documentation therefore always describes the latest
release.

One-time setup:

1. On PyPI, add a trusted publisher for the project `polars-xsum`: the GitHub owner and
   repository, workflow `ci.yml`, environment `pypi`. Before the first release, the project does
   not exist yet; add a *pending* publisher under *Your account → Publishing* instead.
2. In the GitHub repository settings, create the environment `pypi` (optionally with required
   reviewers).
3. In the GitHub repository settings, under *Pages*, set the source to *GitHub Actions*. This
   creates the environment `github-pages`; allow deployments from tags `v*` in its deployment
   rules (by default only the default branch may deploy).
4. Make the repository public before pushing the first release tag. The PyPI project links, the
   GitHub release and the documentation's GitHub link point to it.

## Project layout

```
crates/xsum-core/     pure Rust library: the superaccumulators (no Polars dependency, not published)
  src/ieee.rs           IEEE-754 binary64 layout helpers
  src/small.rs          small superaccumulator (67 × 32-bit digits in i64)
  src/large.rs          large superaccumulator (4096 sign/exponent chunks, lanes)
  src/round.rs          exact → f64 rounding (nearest-even, round-to-odd for f32)
  src/parallel.rs       rayon fold/reduce over blocks
  src/cache.rs          per-thread cache of large accumulators
  tests/                edge cases + proptest against a num-bigint oracle
  benches/sum.rs        criterion benchmarks
crates/polars-xsum/   the Polars plugin (cdylib, not published as a crate)
  src/expressions.rs    plugin entry points and dtype checks
  src/column.rs         chunk/null handling, sums of whole columns and of groups
  src/rolling.rs        rolling_xsum and rolling_xsum_by
  src/pool.rs           the plugin's own thread pool
python/polars_xsum/   Python API: xsum, rolling_xsum, rolling_xsum_by, thread_pool_size
tests/                pytest suite with an exact `fractions.Fraction` oracle
benchmarks/           Python-level benchmarks
docs/                 Sphinx documentation (includes README.md, CONTRIBUTING.md, …)
.github/              CI workflow, Dependabot, release-notes script
```
