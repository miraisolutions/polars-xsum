"""``rolling_xsum``/``rolling_xsum_by`` against Polars and exact rational arithmetic."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Literal

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from polars.testing import assert_frame_equal, assert_series_equal

from polars_xsum import rolling_xsum, rolling_xsum_by

from oracle import expected_sum, same_float

Closed = Literal["left", "right", "both", "none"]


def integer_column(
    n: int, dtype: type[pl.DataType] = pl.Float64, *, nulls: bool = True
) -> pl.Series:
    """Small integers, optionally with nulls: every window sum is exact, also for Polars."""
    values = pl.Series("x", np.random.default_rng(n).integers(-1000, 1000, n), dtype=dtype)
    return values.set(pl.Series(np.arange(n) % 7 == 3), None) if nulls else values


@pytest.mark.parametrize("n", [0, 1, 5, 3 * 2**14 + 5])
@pytest.mark.parametrize(
    ("window_size", "min_samples"),
    # Polars < 2 handles `window_size=0` and `min_samples=0` inconsistently; see
    # test_empty_windows for those.
    [(w, m) for w in (1, 2, 5, 1000) for m in (None, 1, 2) if m is None or m <= w],
)
@pytest.mark.parametrize("center", [False, True])
def test_matches_polars_rolling_sum(
    n: int, window_size: int, min_samples: int | None, *, center: bool
) -> None:
    df = integer_column(n).to_frame()
    out = df.select(
        exact=rolling_xsum("x", window_size, min_samples=min_samples, center=center),
        polars=pl.col("x").rolling_sum(window_size, min_samples=min_samples, center=center),
    )
    assert_series_equal(out["exact"], out["polars"], check_names=False)


def test_empty_windows() -> None:
    df = pl.DataFrame({"x": [1.0, None, None, 4.0]})
    out = df.select(
        zero=rolling_xsum("x", 0),
        min0=rolling_xsum("x", 1, min_samples=0),
        centered=rolling_xsum("x", 2, min_samples=0, center=True),
    )
    assert out["zero"].to_list() == [0.0] * 4
    assert out["min0"].to_list() == [1.0, 0.0, 0.0, 4.0]
    assert out["centered"].to_list() == [1.0, 1.0, 0.0, 4.0]


def test_float32() -> None:
    df = integer_column(100, pl.Float32).to_frame()
    out = df.select(exact=rolling_xsum("x", 4), polars=pl.col("x").rolling_sum(4))
    assert out.schema["exact"] == pl.Float32
    assert_series_equal(out["exact"], out["polars"], check_names=False)


# Nulls, specials, and values with a wide exponent spread.
window_value = st.one_of(
    st.none(), st.floats(), st.integers(0, 2**64 - 1).map(lambda b: math.ldexp(b, -40))
)
window_values = st.lists(window_value)


@given(
    values=window_values,
    window_size=st.integers(0, 6),
    min_samples=st.integers(0, 6),
    center=st.booleans(),
)
def test_matches_exact_rational_sums(
    values: list[float | None], window_size: int, min_samples: int, *, center: bool
) -> None:
    min_samples = min(min_samples, window_size)
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Float64)})
    out = df.select(rolling_xsum("x", window_size, min_samples=min_samples, center=center))
    before = window_size // 2 if center else window_size - 1
    for row, total in enumerate(out["x"].to_list()):
        start = row - before
        window = values[max(0, start) : max(0, start + window_size)]
        present = [v for v in window if v is not None]
        if len(present) < min_samples:
            assert total is None
        else:
            assert same_float(total, expected_sum(present))


def test_over_keeps_windows_within_groups() -> None:
    df = pl.DataFrame({"g": [1, 2, 1, 2, 1, 2], "x": [1e100, 1.0, 1.0, 2.0, -1e100, 3.0]})
    out = df.with_columns(t=rolling_xsum("x", 2).over("g"))
    assert out["t"].to_list() == [None, None, 1e100, 3.0, -1e100, 5.0]


def test_lazy_streaming_and_series_input() -> None:
    df = integer_column(1000).to_frame()
    query = df.lazy().with_columns(t=rolling_xsum("x", 3, center=True))
    assert query.collect_schema()["t"] == pl.Float64
    assert_frame_equal(query.collect(engine="streaming"), query.collect())
    assert_series_equal(
        df.select(rolling_xsum(df["x"], 3)).to_series(),
        df.select(rolling_xsum("x", 3)).to_series(),
    )


def test_chunked_input() -> None:
    first, second = integer_column(50), integer_column(70)
    chunked = pl.concat([first, second], rechunk=False)
    assert chunked.n_chunks() == 2
    out = chunked.to_frame().select(exact=rolling_xsum("x", 5), polars=pl.col("x").rolling_sum(5))
    assert_series_equal(out["exact"], out["polars"], check_names=False)


def test_rejects_non_float_dtypes() -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(pl.exceptions.ComputeError, match="requires a Float32 or Float64 column"):
        df.select(rolling_xsum("x", 2))


@pytest.mark.parametrize(
    ("window_size", "min_samples", "match"),
    [
        (-1, None, "window_size"),
        (2, -1, "min_samples"),
        (2, 3, "min_samples"),
        (2.5, None, "window_size must be an int"),
        (2**64 - 1, None, "below 2\\*\\*63"),
    ],
)
def test_rejects_invalid_windows(window_size: int, min_samples: int | None, match: str) -> None:
    df = pl.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises((TypeError, ValueError, pl.exceptions.ComputeError), match=match):
        df.select(rolling_xsum("x", window_size, min_samples=min_samples))


def keyed_frame(n: int, *, sort: bool) -> pl.DataFrame:
    """Integer-valued ``x`` (exact sums) and integer keys with ties.

    Without nulls: ``rolling_sum_by`` supports them only from Polars 2.0. See
    test_by_nulls and the property test for nulls.
    """
    rng = np.random.default_rng(n + 1)
    keys = pl.Series("t", rng.integers(0, max(1, n // 3), n))
    df = pl.DataFrame({"t": keys, "x": integer_column(n, nulls=False)})
    return df.sort("t") if sort else df


@pytest.mark.parametrize("n", [0, 1, 7, 3 * 2**14 + 5])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("window_size", ["1i", "3i", "1000i"])
@pytest.mark.parametrize("min_samples", [0, 2])
@pytest.mark.parametrize("closed", ["right", "left", "both", "none"])
def test_by_matches_polars_rolling_sum_by(
    n: int, window_size: str, min_samples: int, closed: Closed, *, sort: bool
) -> None:
    df = keyed_frame(n, sort=sort)
    out = df.select(
        exact=rolling_xsum_by("x", "t", window_size, min_samples=min_samples, closed=closed),
        polars=pl.col("x").rolling_sum_by("t", window_size, min_samples=min_samples, closed=closed),
    )
    assert_series_equal(out["exact"], out["polars"], check_names=False)


@pytest.mark.parametrize(
    ("keys", "window_size"),
    [
        (pl.date_range(date(2024, 1, 1), date(2024, 3, 31), "1d", eager=True), "7d"),
        (pl.date_range(date(2024, 1, 31), date(2025, 12, 31), "1mo", eager=True), "1mo"),
        (
            pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "5h", eager=True),
            "1d12h",
        ),
        (
            pl.datetime_range(pl.datetime(2024, 1, 1), pl.datetime(2024, 1, 3), "5h", eager=True),
            timedelta(days=1, hours=2, microseconds=5),
        ),
        (
            # Across the end of daylight saving time: "1d" is a calendar day.
            pl.datetime_range(
                pl.datetime(2024, 10, 25),
                pl.datetime(2024, 10, 30),
                "7h",
                time_unit="ms",
                time_zone="Europe/Zurich",
                eager=True,
            ),
            "1d",
        ),
    ],
)
def test_by_temporal_keys(keys: pl.Series, window_size: str | timedelta) -> None:
    n = len(keys)
    df = pl.DataFrame({"t": keys, "x": integer_column(n, nulls=False)}).reverse()
    out = df.select(
        exact=rolling_xsum_by("x", "t", window_size),
        polars=pl.col("x").rolling_sum_by("t", window_size),
    )
    assert_series_equal(out["exact"], out["polars"], check_names=False)


@pytest.mark.parametrize(
    ("min_samples", "expected"), [(0, [1.0, None, 1.0, 3.0]), (2, [None, None, None, None])]
)
def test_by_nulls(min_samples: int, expected: list[float | None]) -> None:
    df = pl.DataFrame({"t": [1, None, 2, 3], "x": [1.0, 5.0, None, 3.0]})
    out = df.select(rolling_xsum_by("x", "t", "2i", min_samples=min_samples))
    assert out["x"].to_list() == expected


@given(
    rows=st.lists(st.tuples(st.one_of(st.none(), st.integers(0, 8)), window_value)),
    window=st.integers(1, 4),
    min_samples=st.integers(0, 3),
    closed=st.sampled_from(["right", "left", "both", "none"]),
)
def test_by_matches_exact_rational_sums(
    rows: list[tuple[int | None, float | None]], window: int, min_samples: int, closed: Closed
) -> None:
    df = pl.DataFrame(rows, schema={"t": pl.Int64, "x": pl.Float64}, orient="row")
    out = df.select(rolling_xsum_by("x", "t", f"{window}i", min_samples=min_samples, closed=closed))
    include_low, include_high = closed in ("left", "both"), closed in ("right", "both")
    for (key, _), total in zip(rows, out["x"].to_list(), strict=True):
        if key is None:
            assert total is None
            continue
        low = key - window
        present = [
            x
            for t, x in rows
            if t is not None
            and x is not None
            and (low < t or (include_low and t == low))
            and (t < key or (include_high and t == key))
        ]
        if len(present) < min_samples:
            assert total is None
        else:
            assert same_float(total, expected_sum(present))


def test_by_over_lazy_and_streaming() -> None:
    df = keyed_frame(1000, sort=False).with_columns(g=pl.int_range(pl.len()) % 3)
    query = df.lazy().with_columns(
        exact=rolling_xsum_by("x", "t", "5i").over("g"),
        polars=pl.col("x").rolling_sum_by("t", "5i").over("g"),
    )
    assert query.collect_schema()["exact"] == pl.Float64
    out = query.collect()
    assert_series_equal(out["exact"], out["polars"], check_names=False)
    assert_frame_equal(query.collect(engine="streaming"), out)


@pytest.mark.parametrize(
    ("by", "window_size", "kwargs", "match"),
    [
        ("d", "2i", {}, "requires an integer key column"),
        ("t", "2d", {}, "offset_by|Date or Datetime"),
        ("f", "2i", {}, "integer, Date or Datetime key column"),
        ("t", "2i", {"min_samples": -1}, "min_samples"),
        ("t", "2i", {"closed": "sideways"}, "closed"),
        ("t", timedelta(days=-1), {}, "window_size"),
        ("t", timedelta(0), {}, "window_size"),
        ("t", "0i", {}, "window_size"),
        ("t", "-2i", {}, "window_size must be positive"),
        ("t", 2, {}, "window_size must be a str or timedelta"),
        ("t", "2i", {"min_samples": 1.5}, "min_samples must be an int"),
    ],
)
def test_by_rejects_invalid_arguments(
    by: str, window_size: str | timedelta, kwargs: dict, match: str
) -> None:
    df = pl.DataFrame({"t": [1, 2], "d": [date(2024, 1, 1)] * 2, "f": [1.0, 2.0], "x": [1.0, 2.0]})
    with pytest.raises((TypeError, ValueError, pl.exceptions.PolarsError), match=match):
        df.select(rolling_xsum_by("x", by, window_size, **kwargs))


@pytest.mark.parametrize(
    ("closed", "expected"),
    [
        ("right", [1.0, 3.0, 4.0, 12.0]),
        ("left", [0.0, 1.0, 0.0, 4.0]),
        ("both", [1.0, 3.0, 4.0, 12.0]),
        ("none", [0.0, 1.0, 0.0, 4.0]),
    ],
)
def test_by_extreme_integer_keys(closed: Closed, expected: list[float]) -> None:
    # `key - window_size` leaves the Int64 range here.
    lowest, highest = -(2**63), 2**63 - 1
    df = pl.DataFrame({"t": [lowest, lowest + 1, highest - 1, highest], "x": [1.0, 2.0, 4.0, 8.0]})
    out = df.select(rolling_xsum_by("x", "t", "2i", closed=closed))
    assert out["x"].to_list() == expected


def test_by_many_small_groups() -> None:
    df = keyed_frame(20_000, sort=False).with_columns(g=pl.int_range(pl.len()) // 4)
    out = df.select(
        exact=rolling_xsum_by("x", "t", "1000i").over("g"),
        polars=pl.col("x").rolling_sum_by("t", "1000i").over("g"),
    )
    assert_series_equal(out["exact"], out["polars"], check_names=False)


def test_by_float32_chunked_and_series_key() -> None:
    df = keyed_frame(3000, sort=False).with_columns(pl.col("x").cast(pl.Float32))
    chunked = pl.concat([df[:1000], df[1000:]], rechunk=False)
    assert chunked.n_chunks() == 2
    out = chunked.select(
        exact=rolling_xsum_by("x", chunked["t"], "50i"),
        polars=pl.col("x").rolling_sum_by("t", "50i"),
    )
    assert out.schema["exact"] == pl.Float32
    assert_series_equal(out["exact"], out["polars"], check_names=False)
