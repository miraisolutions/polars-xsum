"""Special values, overflow, underflow, rounding ties, nulls and chunking."""

from __future__ import annotations

import math
import sys

import polars as pl
import pytest

from polars_xsum import xsum

from oracle import expected_sum, float_bits, float_from_bits, run_xsum, same_float

MAX = sys.float_info.max
TINY = 5e-324  # smallest subnormal double
EPS = sys.float_info.epsilon
HALF_ULP_OF_MAX = 2.0**970
F32_MAX = 3.4028234663852886e38

CASES_F64: list[tuple[str, list[float | None], float]] = [
    ("empty", [], 0.0),
    ("negative zero", [-0.0], 0.0),
    ("negative zeros", [-0.0, -0.0], 0.0),
    ("exact cancellation", [1.5, -1.5], 0.0),
    ("nan", [math.nan], math.nan),
    ("nan among numbers", [1.0, math.nan, 2.0], math.nan),
    ("nan and infinity", [math.nan, math.inf], math.nan),
    ("opposite infinities", [math.inf, -math.inf], math.nan),
    ("opposite infinities apart", [-math.inf, 1.0, math.inf], math.nan),
    ("positive infinity", [math.inf, 1.0], math.inf),
    ("negative infinity", [-1.0, -math.inf, -MAX], -math.inf),
    ("infinity is not cancelled", [math.inf, -MAX, -MAX], math.inf),
    ("overflow", [MAX, MAX], math.inf),
    ("negative overflow", [-MAX, -MAX], -math.inf),
    ("intermediate overflow", [MAX, MAX, -MAX], MAX),
    ("intermediate negative overflow", [-MAX, -MAX, MAX, 1.0], -MAX),
    ("tie at overflow threshold", [MAX, HALF_ULP_OF_MAX], math.inf),
    ("negative tie at overflow threshold", [-MAX, -HALF_ULP_OF_MAX], -math.inf),
    ("just below overflow threshold", [MAX, HALF_ULP_OF_MAX, -TINY], MAX),
    ("subnormals", [TINY] * 7, 7 * TINY),
    ("subnormal cancellation", [TINY, -TINY], 0.0),
    ("subnormals into normal range", [2.0**-1022 - TINY, TINY], 2.0**-1022),
    ("normal into subnormal range", [2.0**-1022, -(2.0**-1022 - TINY)], TINY),
    ("tiny survives huge cancellation", [1e300, TINY, -1e300], TINY),
    ("underflowing difference", [1.0 + EPS, -1.0, -EPS, 1e-300], 1e-300),
    ("catastrophic cancellation", [1.0, 1e100, 1.0, -1e100], 2.0),
    ("cancellation near max", [1e308, 1.0, -1e308, 1e-308], 1.0),
    ("tenths", [0.1] * 10, 1.0),
    ("0.1 + 0.2 + 0.3", [0.1, 0.2, 0.3], 0.6),
    ("tie to even (down)", [1.0, EPS / 2], 1.0),
    ("broken tie", [1.0, EPS / 2, 1e-300], 1.0 + EPS),
    ("tie to even (up)", [1.0, EPS, EPS / 2], 1.0 + 2 * EPS),
    ("negative tie to even", [-1.0, -EPS, -EPS / 2], -1.0 - 2 * EPS),
    ("nulls are skipped", [None, 1e100, None, 1.0, -1e100, None], 1.0),
    ("all null", [None, None], 0.0),
    ("nulls with nan", [None, math.nan], math.nan),
]


@pytest.mark.parametrize(
    ("values", "expected"),
    [pytest.param(values, expected, id=name) for name, values, expected in CASES_F64],
)
def test_float64_edge_cases(values: list[float | None], expected: float) -> None:
    result = run_xsum(values)
    assert same_float(result, expected), f"{result!r} != {expected!r}"
    assert same_float(expected, expected_sum(values)), "the oracle disagrees with the case"


CASES_F32: list[tuple[str, list[float | None], float]] = [
    ("empty", [], 0.0),
    ("negative zeros", [-0.0, -0.0], 0.0),
    ("nan", [1.0, math.nan], math.nan),
    ("opposite infinities", [math.inf, -math.inf], math.nan),
    ("infinity", [-math.inf, F32_MAX], -math.inf),
    ("overflow", [F32_MAX, F32_MAX], math.inf),
    ("intermediate overflow", [F32_MAX, F32_MAX, -F32_MAX], F32_MAX),
    ("tie at overflow threshold", [F32_MAX, 2.0**103], math.inf),
    ("just below overflow threshold", [F32_MAX, 2.0**103, -(2.0**-149)], F32_MAX),
    ("subnormals", [2.0**-149] * 3, 3 * 2.0**-149),
    # Rounding the exact sum to f64 first and then to f32 would give 1.0.
    ("no double rounding", [1.0, 2.0**-24, 2.0**-60], 1.0 + 2.0**-23),
    ("exact tie", [1.0, 2.0**-24], 1.0),
    ("nulls are skipped", [None, 1e30, 1.0, -1e30, None], 1.0),
]


@pytest.mark.parametrize(
    ("values", "expected"),
    [pytest.param(values, expected, id=name) for name, values, expected in CASES_F32],
)
def test_float32_edge_cases(values: list[float | None], expected: float) -> None:
    result = run_xsum(values, pl.Float32)
    assert same_float(result, expected), f"{result!r} != {expected!r}"
    assert same_float(expected, expected_sum(values, single=True))


def test_nan_payload_of_first_nan_is_kept() -> None:
    first = float_from_bits(0x7FF8_0000_0000_BEEF)
    second = float_from_bits(0x7FF8_0000_0000_1234)
    for n in [10, 5_000, 100_000]:
        values = [1.0] * n
        values[3], values[n - 2] = first, second
        assert float_bits(run_xsum(values)) == float_bits(first)


def test_nan_payload_is_kept_after_opposite_infinities() -> None:
    payload = float_from_bits(0x7FF8_0000_0000_BEEF)
    for n in [10, 5_000, 100_000]:
        values = [1.0] * n
        values[1], values[n // 2], values[n - 1] = math.inf, -math.inf, payload
        assert float_bits(run_xsum(values)) == float_bits(payload)


def test_multiple_chunks() -> None:
    parts = [
        pl.Series("x", [1e100, None, 1.0]),
        pl.Series("x", [-1e100, 2.0]),
        pl.Series("x", [None, 0.5]),
    ]
    series = pl.concat(parts, rechunk=False)
    assert series.n_chunks() == 3
    assert series.to_frame().select(xsum("x")).item() == 3.5


def test_sliced_series_with_validity_offset() -> None:
    values = [None if i % 3 == 0 else float(i) * 0.1 for i in range(500)]
    series = pl.Series("x", values)
    for offset, length in [(1, 499), (5, 64), (63, 130), (64, 64), (100, 1), (499, 1)]:
        sliced = series.slice(offset, length)
        expected = expected_sum(values[offset : offset + length])
        assert same_float(sliced.to_frame().select(xsum("x")).item(), expected)


def test_null_slots_do_not_leak_their_values() -> None:
    # `set` with a mask creates nulls whose underlying buffer slots still hold data.
    series = pl.Series("x", [1.0, math.nan, 2.0, math.inf] * 50)
    masked = series.set(series.is_nan() | series.is_infinite(), None)
    assert masked.null_count() == 100
    assert masked.to_frame().select(xsum("x")).item() == 150.0


def test_long_input_with_known_sum_uses_all_code_paths() -> None:
    # Each triple sums to exactly 1, but naive summation loses every 1.0.
    for triples in [10, 1_000, 100_000, 1_000_000]:
        df = pl.DataFrame({"x": [1e16, 1.0, -1e16] * triples})
        out = df.select(naive=pl.col("x").sum(), exact=xsum("x"))
        assert out.row(0) == (0.0, float(triples))


def test_long_input_with_nulls_and_chunks() -> None:
    n = 1_000_003
    values = pl.Series("x", [0.1, None, -0.3, 0.7e-20, None, 1e10]).sample(
        n, with_replacement=True, seed=1
    )
    chunked = pl.concat([values.slice(0, n // 3), values.slice(n // 3)], rechunk=False)
    expected = expected_sum(values.to_list())
    assert same_float(values.to_frame().select(xsum("x")).item(), expected)
    assert same_float(chunked.to_frame().select(xsum("x")).item(), expected)


def test_long_float32_input() -> None:
    values = pl.Series("x", [0.1, -0.2, 3.0e-5, 7.0], dtype=pl.Float32).sample(
        500_000, with_replacement=True, seed=7
    )
    expected = expected_sum(values.to_list(), single=True)
    assert same_float(values.to_frame().select(xsum("x")).item(), expected)
