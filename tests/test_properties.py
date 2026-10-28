"""Property-based tests against exact rational arithmetic."""

from __future__ import annotations

import itertools
import math
from typing import TYPE_CHECKING

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from polars_xsum import xsum

from oracle import expected_sum, float_from_bits, run_xsum, same_float

if TYPE_CHECKING:
    import random

    from hypothesis.strategies import SearchStrategy

# Any double, including subnormals, NaN and both infinities.
any_f64 = st.floats(allow_nan=True, allow_infinity=True)
finite_f64 = st.floats(allow_nan=False, allow_infinity=False)
# Doubles drawn uniformly from all bit patterns: wide exponent spread.
bit_pattern_f64 = st.integers(0, 2**64 - 1).map(float_from_bits)
# Doubles with nearby exponents, so they share accumulator chunks.
clustered_f64 = st.floats(min_value=-1e3, max_value=1e3, allow_subnormal=False)
mixed_f64 = st.one_of(any_f64, bit_pattern_f64, clustered_f64)

any_f32 = st.floats(width=32, allow_nan=True, allow_infinity=True)


def with_nulls(elements: SearchStrategy[float]) -> SearchStrategy[list[float | None]]:
    return st.lists(st.one_of(st.none(), elements), max_size=200)


@st.composite
def cancelling(draw: st.DrawFn) -> list[float]:
    """Values and their negations in random order, plus a few small extras."""
    base = draw(st.lists(finite_f64, max_size=100))
    extras = draw(st.lists(st.floats(-1.0, 1.0), max_size=3))
    values = [v for x in base for v in (x, -x)] + extras
    return draw(st.permutations(values))


@given(with_nulls(mixed_f64))
def test_float64_matches_exact_rational_sum(values: list[float | None]) -> None:
    assert same_float(run_xsum(values), expected_sum(values))


@given(with_nulls(any_f32))
def test_float32_matches_exact_rational_sum(values: list[float | None]) -> None:
    assert same_float(run_xsum(values, pl.Float32), expected_sum(values, single=True))


@given(cancelling())
def test_cancellation(values: list[float]) -> None:
    assert same_float(run_xsum(values), expected_sum(values))


@given(st.lists(finite_f64, max_size=100))
def test_agrees_with_fsum_when_fsum_does_not_overflow(values: list[float]) -> None:
    try:
        reference = math.fsum(values)
    except OverflowError:  # fsum fails on intermediate overflow; xsum does not
        return
    if math.isfinite(reference):
        assert run_xsum(values) == reference


@given(st.lists(mixed_f64, max_size=100), st.randoms(use_true_random=False))
def test_order_does_not_matter(values: list[float], rng: random.Random) -> None:
    shuffled = values.copy()
    rng.shuffle(shuffled)
    assert same_float(run_xsum(values), run_xsum(shuffled))


@given(with_nulls(mixed_f64), st.lists(st.integers(0, 200), max_size=4))
def test_chunking_does_not_matter(values: list[float | None], cuts: list[int]) -> None:
    bounds = sorted({min(c, len(values)) for c in cuts} | {0, len(values)})
    parts = [
        pl.Series("x", values[lo:hi], dtype=pl.Float64) for lo, hi in itertools.pairwise(bounds)
    ] or [pl.Series("x", [], dtype=pl.Float64)]
    chunked = pl.concat(parts, rechunk=False)
    assert same_float(chunked.to_frame().select(xsum("x")).item(), run_xsum(values))


@settings(max_examples=25, deadline=None)
@given(
    st.lists(mixed_f64, min_size=1, max_size=50),
    st.integers(40, 2_000),
    st.randoms(use_true_random=False),
)
def test_long_inputs(base: list[float], repeats: int, rng: random.Random) -> None:
    # Repeating a few values many times exercises chunk rollover in the large
    # accumulator, and the parallel path for the longest inputs.
    values = base * repeats
    rng.shuffle(values)
    assert same_float(run_xsum(values), expected_sum(values))


@settings(max_examples=30)
@given(st.lists(st.tuples(st.integers(0, 4), with_nulls(clustered_f64)), min_size=1, max_size=8))
def test_group_by_matches_per_group_sums(groups: list[tuple[int, list[float | None]]]) -> None:
    keys = [key for key, values in groups for _ in values]
    data = [v for _, values in groups for v in values]
    df = pl.DataFrame({"g": keys, "x": data}, schema={"g": pl.Int64, "x": pl.Float64})
    out = dict(df.group_by("g").agg(xsum("x")).iter_rows())
    for key in set(keys):
        expected = expected_sum([v for k, v in zip(keys, data, strict=True) if k == key])
        assert same_float(out[key], expected)
