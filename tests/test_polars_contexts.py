"""``xsum`` in aggregation, window, lazy and streaming contexts."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from polars_xsum import xsum

from oracle import expected_sum, same_float


@pytest.fixture
def grouped() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "g": ["a", "b", "a", "c", "b", "a", "d", "d"],
            "x": [1e100, 2.0, 1.0, None, math.inf, -1e100, None, None],
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )


def expected_per_group(df: pl.DataFrame, column: str) -> dict[str, float]:
    groups: dict[str, list[float | None]] = {}
    for key, value in df.select("g", column).iter_rows():
        groups.setdefault(key, []).append(value)
    return {key: expected_sum(values) for key, values in groups.items()}


def test_group_by_agg(grouped: pl.DataFrame) -> None:
    out = grouped.group_by("g", maintain_order=True).agg(xsum("x"), xsum("y"))
    assert out.schema == pl.Schema({"g": pl.String, "x": pl.Float64, "y": pl.Float64})
    expected_x = expected_per_group(grouped, "x")
    expected_y = expected_per_group(grouped, "y")
    for key, x, y in out.iter_rows():
        assert same_float(x, expected_x[key])
        assert same_float(y, expected_y[key])
    assert out["x"].to_list() == [1.0, math.inf, 0.0, 0.0]


def test_group_by_many_groups() -> None:
    n = 200_000
    df = pl.DataFrame(
        {"g": pl.int_range(n, eager=True) % 1000, "x": [1e16, 1.0, -1e16, 0.25] * (n // 4)}
    )
    out = df.group_by("g").agg(xsum("x"), naive=pl.col("x").sum()).sort("g")
    expected = expected_per_group(df.with_columns(pl.col("g").cast(pl.String)), "x")
    for key, exact, _ in out.iter_rows():
        assert same_float(exact, expected[str(key)])


def test_over(grouped: pl.DataFrame) -> None:
    out = grouped.with_columns(total=xsum("x").over("g"))
    expected = expected_per_group(grouped, "x")
    for key, total in out.select("g", "total").iter_rows():
        assert same_float(total, expected[key])


def test_lazy_pipeline_and_streaming(grouped: pl.DataFrame) -> None:
    query = (
        grouped.lazy()
        .filter(pl.col("x").is_finite())
        .group_by("g")
        .agg(xsum("x"), n=pl.len())
        .sort("g")
    )
    in_memory = query.collect()
    assert_frame_equal(query.collect(engine="streaming"), in_memory)
    assert in_memory["x"].to_list() == [1.0, 2.0]


def test_select_multiple_and_aliases() -> None:
    df = pl.DataFrame({"a": [0.1] * 10, "b": [1e308, 1e308, -1e308] + [0.0] * 7})
    out = df.select(xsum("a").alias("sum_a"), xsum("b").alias("sum_b"))
    assert out.row(0) == (1.0, 1e308)


def test_group_by_float32() -> None:
    df = pl.DataFrame(
        {"g": [1, 1, 2], "x": [1.0, 2.0**-24, 2.0**-60]},
        schema={"g": pl.Int64, "x": pl.Float32},
    )
    out = df.group_by("g", maintain_order=True).agg(xsum("x"))
    assert out.schema["x"] == pl.Float32
    assert out["x"].to_list() == [1.0, 2.0**-60]


def test_nested_expressions_in_agg(grouped: pl.DataFrame) -> None:
    out = grouped.group_by("g", maintain_order=True).agg(
        filtered=xsum(pl.col("y").filter(pl.col("y") > 0.25)),
        scaled=xsum(pl.col("y") * 10.0),
        empty=xsum(pl.col("y").filter(pl.col("y") > 1.0)),
    )
    assert out["filtered"].to_list() == [0.3 + 0.6, 0.5, 0.4, 0.7 + 0.8]
    assert out["scaled"].to_list() == [10.0, 7.0, 4.0, 15.0]
    assert out["empty"].to_list() == [0.0] * 4


def test_over_with_sorted_and_multiple_keys() -> None:
    df = pl.DataFrame(
        {
            "k1": [1, 1, 1, 2, 2, 2],
            "k2": ["a", "b", "a", "a", "a", None],
            "x": [1e100, 2.0, -1e100, 0.1, 0.2, 0.3],
        }
    )
    out = df.with_columns(t1=xsum("x").over("k1"), t2=xsum("x").over("k1", "k2"))
    assert out["t1"].to_list() == [2.0] * 3 + [expected_sum([0.1, 0.2, 0.3])] * 3
    pair = expected_sum([0.1, 0.2])
    assert out["t2"].to_list() == [0.0, 2.0, 0.0, pair, pair, 0.3]


def test_rolling() -> None:
    values = [1e100, 1.0, -1e100, 0.1, None, 0.2, 0.3, math.inf, 1.0]
    df = pl.DataFrame({"t": range(len(values)), "x": values})
    out = df.rolling("t", period="3i").agg(xsum("x"))
    expected = [expected_sum(values[max(0, i - 2) : i + 1]) for i in range(len(values))]
    assert all(map(same_float, out["x"].to_list(), expected))
    windowed = df.with_columns(w=xsum("x").rolling("t", period="3i"))
    assert all(map(same_float, windowed["w"].to_list(), expected))


def test_group_by_dynamic() -> None:
    values = [1e100, 1.0, -1e100, 0.1, None, 0.2, 0.3]
    df = pl.DataFrame({"t": range(len(values)), "x": values})
    out = df.group_by_dynamic("t", every="2i", period="4i").agg(xsum("x"))
    expected = [expected_sum(values[start : start + 4]) for start in out["t"].to_list()]
    assert all(map(same_float, out["x"].to_list(), expected))
