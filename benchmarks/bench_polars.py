"""Benchmark ``xsum`` and the rolling sums against Polars' built-in sum, NumPy and ``math.fsum``.

Run with::

    uv run python benchmarks/bench_polars.py [--max-exp 8] [--repeats 7]

The script prints Markdown tables (used in ``BENCHMARKS.md``).
"""

from __future__ import annotations

import argparse
import math
import os
import platform
import statistics
import time
from fractions import Fraction
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

import polars_xsum
from polars_xsum import rolling_xsum, rolling_xsum_by, xsum

if TYPE_CHECKING:
    from collections.abc import Callable


def median_seconds(func: Callable[[], object], repeats: int) -> float:
    """Return the median wall time of ``repeats`` calls (after one warm-up)."""
    func()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def group_sum(df: pl.DataFrame, expr: pl.Expr) -> pl.DataFrame:
    """Aggregate ``expr`` per group ``g``."""
    return df.group_by("g").agg(expr)


def fmt_time(seconds: float) -> str:
    """Format a duration with a sensible unit."""
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def bench_sizes(max_exp: int, repeats: int) -> None:
    """Whole-column sums of normally distributed Float64 data."""
    rng = np.random.default_rng(42)
    print("### Whole-column sum, Float64, N(0, 1000²)\n")
    print("| n | `pl.sum` | `xsum` | `np.sum` | `math.fsum` | xsum ns/elem | xsum GB/s |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for exp in range(3, max_exp + 1):
        n = 10**exp
        values = rng.normal(0.0, 1000.0, n)
        df = pl.DataFrame({"x": values})
        t_polars = median_seconds(partial(df.select, pl.col("x").sum()), repeats)
        t_xsum = median_seconds(partial(df.select, xsum("x")), repeats)
        t_numpy = median_seconds(partial(np.sum, values), repeats)
        fsum = (
            fmt_time(median_seconds(partial(math.fsum, values), max(1, repeats // 3)))
            if n <= 10**7
            else "n/a"
        )
        print(
            f"| 10^{exp} | {fmt_time(t_polars)} | {fmt_time(t_xsum)} | {fmt_time(t_numpy)} "
            f"| {fsum} | {t_xsum / n * 1e9:.3f} | {8 * n / t_xsum / 1e9:.1f} |"
        )
    print()


def bench_float32(repeats: int) -> None:
    """Whole-column sums of Float32 data."""
    rng = np.random.default_rng(7)
    print("### Whole-column sum, Float32\n")
    print("| n | `pl.sum` | `xsum` |")
    print("|---:|---:|---:|")
    for n in (10**4, 10**6, 10**7):
        df = pl.DataFrame({"x": rng.normal(0.0, 1.0, n).astype(np.float32)})
        t_polars = median_seconds(partial(df.select, pl.col("x").sum()), repeats)
        t_xsum = median_seconds(partial(df.select, xsum("x")), repeats)
        print(f"| {n:.0e} | {fmt_time(t_polars)} | {fmt_time(t_xsum)} |")
    print()


def bench_nulls(repeats: int) -> None:
    """Whole-column sums with 10% nulls."""
    rng = np.random.default_rng(3)
    n = 10**7
    values = pl.Series("x", rng.normal(0.0, 1.0, n))
    df = pl.DataFrame({"x": values.set(pl.Series(rng.random(n) < 0.1), None)})
    t_polars = median_seconds(partial(df.select, pl.col("x").sum()), repeats)
    t_xsum = median_seconds(partial(df.select, xsum("x")), repeats)
    print("### Whole-column sum with 10% nulls, n = 10^7\n")
    print("| `pl.sum` | `xsum` |")
    print("|---:|---:|")
    print(f"| {fmt_time(t_polars)} | {fmt_time(t_xsum)} |\n")


def bench_chunked(repeats: int) -> None:
    """Whole-column sums of a column in 100 chunks."""
    rng = np.random.default_rng(6)
    n = 10**7
    chunks = [pl.Series("x", rng.normal(0.0, 1.0, n // 100)) for _ in range(100)]
    df = pl.concat(chunks, rechunk=False).to_frame()
    t_polars = median_seconds(partial(df.select, pl.col("x").sum()), repeats)
    t_xsum = median_seconds(partial(df.select, xsum("x")), repeats)
    print(f"### Whole-column sum in {df['x'].n_chunks()} chunks, n = 10^7\n")
    print("| `pl.sum` | `xsum` |")
    print("|---:|---:|")
    print(f"| {fmt_time(t_polars)} | {fmt_time(t_xsum)} |\n")


def bench_group_by(repeats: int) -> None:
    """``group_by().agg()`` and ``over()`` with different numbers of groups."""
    rng = np.random.default_rng(1)
    n = 10**7
    print(f"### `group_by(...).agg(...)` and `over(...)`, n = {n:.0e}\n")
    print(
        "| groups | keys | `pl.sum` | `sort().sum()` | `xsum` | ratio "
        "| `sum().over` | `xsum().over` | ratio |"
    )
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for groups in (10, 1_000, 100_000, 1_000_000):
        df = pl.DataFrame({"g": rng.integers(0, groups, n), "x": rng.normal(0.0, 1.0, n)})
        for keys, frame in (("random", df), ("sorted", df.sort("g"))):
            t_polars = median_seconds(partial(group_sum, frame, pl.col("x").sum()), repeats)
            t_sorted = median_seconds(
                partial(group_sum, frame, pl.col("x").sort().sum()), max(1, repeats // 2)
            )
            t_xsum = median_seconds(partial(group_sum, frame, xsum("x")), repeats)
            over = partial(frame.with_columns, t=pl.col("x").sum().over("g"))
            t_over = median_seconds(over, repeats)
            t_xover = median_seconds(partial(frame.with_columns, t=xsum("x").over("g")), repeats)
            print(
                f"| {groups:,} | {keys} | {fmt_time(t_polars)} | {fmt_time(t_sorted)} "
                f"| {fmt_time(t_xsum)} | {t_xsum / t_polars:.1f}x | {fmt_time(t_over)} "
                f"| {fmt_time(t_xover)} | {t_xover / t_over:.1f}x |"
            )
    print()


def bench_rolling(repeats: int) -> None:
    """``rolling_xsum`` against ``rolling_sum`` and ``xsum`` in a ``rolling`` context."""
    rng = np.random.default_rng(2)
    print("### Rolling sums\n")
    print("| n | window | `rolling_sum` | `rolling_xsum` | ratio | `xsum` in `rolling(...)` |")
    print("|---:|---:|---:|---:|---:|---:|")
    for n in (10**6, 10**7):
        df = pl.DataFrame({"t": np.arange(n), "x": rng.normal(0.0, 1.0, n)})
        for window in (10, 1_000, 100_000):
            t_polars = median_seconds(partial(df.select, pl.col("x").rolling_sum(window)), repeats)
            t_xsum = median_seconds(partial(df.select, rolling_xsum("x", window)), repeats)
            # `xsum` in `rolling(...)` gathers every window: n * window values.
            context = (
                fmt_time(
                    median_seconds(partial(df.rolling("t", period=f"{window}i").agg, xsum("x")), 1)
                )
                if n * window <= 10**9
                else "n/a"
            )
            print(
                f"| 10^{round(math.log10(n))} | {window:,} | {fmt_time(t_polars)} "
                f"| {fmt_time(t_xsum)} | {t_xsum / t_polars:.1f}x | {context} |"
            )
    print()


def bench_rolling_by(repeats: int) -> None:
    """``rolling_xsum_by`` against ``rolling_sum_by``, one row per minute on average."""
    rng = np.random.default_rng(4)
    print("### Rolling sums over a `Datetime` key\n")
    print("| n | keys | window | `rolling_sum_by` | `rolling_xsum_by` | ratio |")
    print("|---:|---|---:|---:|---:|---:|")
    for n in (10**6, 10**7):
        minutes = pl.Series("t", np.sort(rng.integers(0, n, n)) * 60_000_000)
        df = pl.DataFrame({"t": minutes.cast(pl.Datetime("us")), "x": rng.normal(0.0, 1.0, n)})
        shuffled = df.sample(fraction=1.0, shuffle=True, seed=1)
        for keys, frame in (("sorted", df), ("unsorted", shuffled)):
            for window in ("1h", "7d"):
                polars = pl.col("x").rolling_sum_by("t", window)
                t_polars = median_seconds(partial(frame.select, polars), repeats)
                exact = rolling_xsum_by("x", "t", window)
                t_xsum = median_seconds(partial(frame.select, exact), repeats)
                print(
                    f"| 10^{round(math.log10(n))} | {keys} | {window} | {fmt_time(t_polars)} "
                    f"| {fmt_time(t_xsum)} | {t_xsum / t_polars:.1f}x |"
                )
    print()


def accuracy() -> None:
    """Error of inexact methods on ill-conditioned data."""
    rng = np.random.default_rng(5)
    n = 10**6
    big = rng.normal(0.0, 1e12, n // 2)
    values = np.concatenate([big, -big, rng.normal(0.0, 1.0, n // 100)])
    rng.shuffle(values)
    exact = float(sum((Fraction(v) for v in values.tolist()), Fraction(0)))
    df = pl.DataFrame({"x": values})
    results = {
        "`pl.sum`": df.select(pl.col("x").sum()).item(),
        "`np.sum` (pairwise)": float(np.sum(values)),
        "`math.fsum`": math.fsum(values),
        "`xsum`": df.select(xsum("x")).item(),
    }
    print(f"### Accuracy: ±N(0, 10¹²) pairs plus {n // 100:,} N(0, 1) values\n")
    print(f"Exact sum (correctly rounded): `{exact!r}`\n")
    print("| method | result | relative error |")
    print("|---|---:|---:|")
    for name, result in results.items():
        error = abs(result - exact) / abs(exact)
        print(f"| {name} | `{result!r}` | {error:.2e} |")
    print()


def main() -> None:
    """Run all benchmarks."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-exp", type=int, default=8, help="largest size is 10^max_exp")
    parser.add_argument("--repeats", type=int, default=7, help="timed repetitions per case")
    args = parser.parse_args()

    print(
        f"polars {pl.__version__}, polars-xsum {polars_xsum.__version__}, numpy {np.__version__}, "
        f"Python {platform.python_version()}, {os.cpu_count()} logical CPUs\n"
    )
    accuracy()
    bench_sizes(args.max_exp, args.repeats)
    bench_float32(args.repeats)
    bench_nulls(args.repeats)
    bench_chunked(args.repeats)
    bench_group_by(args.repeats)
    bench_rolling(args.repeats)
    bench_rolling_by(args.repeats)


if __name__ == "__main__":
    main()
