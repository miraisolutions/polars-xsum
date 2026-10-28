"""Exact reference implementations (based on ``fractions``) and test helpers."""

from __future__ import annotations

import math
import struct
from fractions import Fraction
from typing import TYPE_CHECKING

import polars as pl

from polars_xsum import xsum

if TYPE_CHECKING:
    from collections.abc import Sequence

# The smallest power of two beyond the largest finite float32.
F32_OVERFLOW = Fraction(2**128)


def run_xsum(values: Sequence[float | None], dtype: type[pl.DataType] = pl.Float64) -> float:
    """Sum ``values`` with the plugin and return the scalar result."""
    frame = pl.DataFrame({"x": pl.Series(values, dtype=dtype)})
    result = frame.select(xsum("x")).item()
    assert isinstance(result, float)
    return result


def _special(values: Sequence[float]) -> float | None:
    """Return the IEEE-754 result if ``values`` contain infinities or NaNs."""
    has_pos = math.inf in values
    has_neg = -math.inf in values
    if any(math.isnan(v) for v in values) or (has_pos and has_neg):
        return math.nan
    if has_pos:
        return math.inf
    if has_neg:
        return -math.inf
    return None


def round_to_f64(exact: Fraction) -> float:
    """Round an exact rational to the nearest double (ties to even)."""
    try:
        # int / int true division is correctly rounded in CPython.
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def round_to_f32(exact: Fraction) -> float:
    """Round an exact rational to the nearest float32 (ties to even)."""
    if exact == 0:
        return 0.0
    magnitude = abs(exact)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if Fraction(2) ** exponent > magnitude:
        exponent -= 1
    # 24 significant bits, but never finer than the subnormal spacing 2^-149.
    quantum = Fraction(2) ** max(exponent - 23, -149)
    rounded = round(magnitude / quantum) * quantum  # round() on Fraction: ties to even
    result = math.inf if rounded >= F32_OVERFLOW else float(rounded)
    return result if exact > 0 else -result


def expected_sum(values: Sequence[float | None], *, single: bool = False) -> float:
    """The correctly rounded exact sum of the non-null ``values``."""
    present = [v for v in values if v is not None]
    special = _special(present)
    if special is not None:
        return special
    exact = sum((Fraction(v) for v in present), Fraction(0))
    return round_to_f32(exact) if single else round_to_f64(exact)


def float_bits(value: float) -> int:
    """The IEEE-754 bit pattern of a double."""
    bits: int = struct.unpack("<Q", struct.pack("<d", value))[0]
    return bits


def float_from_bits(bits: int) -> float:
    """The double with the given IEEE-754 bit pattern."""
    value: float = struct.unpack("<d", struct.pack("<Q", bits))[0]
    return value


def same_float(a: float, b: float) -> bool:
    """Bitwise equality, except that all NaNs are considered equal."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return float_bits(a) == float_bits(b)
