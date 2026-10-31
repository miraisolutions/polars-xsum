# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

When releasing, rename the `Unreleased` section to `[X.Y.Z] - YYYY-MM-DD`. The release workflow
refuses to publish a tag without such a section, and uses it as the notes of the GitHub release.

## [Unreleased]

First release.

### Added

- `xsum(expr)`: the exact sum of a `Float32` or `Float64` column, correctly rounded once to the
  column's type. It works in `select`, `with_columns`, `group_by(...).agg(...)`, `over(...)`, and
  in lazy and streaming queries. In `group_by` and `over`, a single plugin call sums all groups.
- `rolling_xsum(expr, window_size, *, min_samples=None, center=False)`: the exact counterpart of
  `Expr.rolling_sum`.
- `rolling_xsum_by(expr, by, window_size, *, min_samples=0, closed="right")`: the exact
  counterpart of `Expr.rolling_sum_by`, for integer, `Date` and `Datetime` keys, which need not be
  sorted.
- `thread_pool_size()`: the size of the plugin's own thread pool, which follows
  `POLARS_MAX_THREADS` (overridable with `RAYON_NUM_THREADS`).
- abi3 wheels for Python ≥ 3.11 on Linux (x86_64 and aarch64), Windows (x64) and macOS (x86_64
  and arm64). Requires Polars ≥ 1.28.
