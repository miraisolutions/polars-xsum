"""Basic behaviour of the ``xsum`` expression."""

from __future__ import annotations

import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import polars_xsum
from polars_xsum import xsum


def test_version_is_exposed() -> None:
    assert isinstance(polars_xsum.__version__, str)
    assert polars_xsum.__version__.count(".") == 2


def test_returns_an_expression() -> None:
    assert isinstance(xsum("x"), pl.Expr)
    assert isinstance(xsum(pl.col("x")), pl.Expr)


def test_exact_where_naive_sum_is_not() -> None:
    df = pl.DataFrame({"x": [1e100, 1.0, -1e100]})
    out = df.select(naive=pl.col("x").sum(), exact=xsum("x"))
    assert out.row(0) == (0.0, 1.0)


def test_column_name_and_expression_input_are_equivalent() -> None:
    df = pl.DataFrame({"x": [0.1] * 10})
    assert_frame_equal(df.select(xsum("x")), df.select(xsum(pl.col("x"))))
    assert df.select(xsum("x")).item() == 1.0


def test_accepts_a_series() -> None:
    series = pl.Series("v", [1e100, 1.0, -1e100])
    assert pl.select(xsum(series)).item() == 1.0


def test_output_keeps_name_and_dtype() -> None:
    df = pl.DataFrame(
        {"a": [1.5, 2.5], "b": [1.5, 2.5]},
        schema={"a": pl.Float64, "b": pl.Float32},
    )
    out = df.select(xsum("a"), xsum("b"), total=xsum(pl.col("a") * 2))
    assert out.schema == pl.Schema({"a": pl.Float64, "b": pl.Float32, "total": pl.Float64})
    assert out.row(0) == (4.0, 4.0, 8.0)


def test_lazy_and_eager_agree() -> None:
    df = pl.DataFrame({"x": [1e16, 1.0, -1e16, 3.5, None]})
    eager = df.select(xsum("x"))
    assert_frame_equal(df.lazy().select(xsum("x")).collect(), eager)
    assert_frame_equal(df.lazy().select(xsum("x")).collect(engine="streaming"), eager)
    assert eager.item() == 4.5


def test_lazy_schema_is_resolved_without_execution() -> None:
    lf = pl.LazyFrame({"x": [1.0]}, schema={"x": pl.Float32})
    assert lf.select(xsum("x")).collect_schema() == pl.Schema({"x": pl.Float32})


def test_broadcasts_in_with_columns() -> None:
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0]})
    out = df.with_columns(share=pl.col("x") / xsum("x"))
    assert out["share"].to_list() == [1 / 6, 2 / 6, 3 / 6]


def test_composes_with_other_expressions() -> None:
    df = pl.DataFrame({"x": [1e308, 1e308, -1e308]})
    assert df.select(xsum("x") * 0.5).item() == 5e307
    assert df.select(xsum(pl.col("x").filter(pl.col("x") > 0))).item() == math.inf


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int32, pl.Int64, pl.UInt64, pl.Decimal(10, 2), pl.String, pl.Boolean, pl.Null],
)
def test_rejects_non_float_dtypes(dtype: pl.DataType) -> None:
    df = pl.DataFrame({"x": pl.Series([None, None], dtype=dtype)})
    with pytest.raises(pl.exceptions.ComputeError, match="requires a Float32 or Float64 column"):
        df.select(xsum("x"))


def test_explicit_cast_allows_integers() -> None:
    df = pl.DataFrame({"x": [2**53, 1, 1]})
    # The cast is lossless here, and the exact sum 2^53 + 2 is representable.
    assert df.select(xsum(pl.col("x").cast(pl.Float64))).item() == 2.0**53 + 2
    assert df.select(pl.col("x").cast(pl.Float64).sum()).item() == 2.0**53
