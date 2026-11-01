"""Exact, correctly rounded summation of floating-point columns for Polars.

The :func:`xsum` expression computes the *exact* sum of a ``Float32`` or
``Float64`` column and rounds it once to the column's type;
:func:`rolling_xsum` and :func:`rolling_xsum_by` do the same for every window
of a fixed number of rows or of a key column. It uses Radford M.
Neal's small and large superaccumulators (arXiv:1505.05571) and is implemented
in Rust.
"""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_xsum._internal import __version__, thread_pool_size

if TYPE_CHECKING:
    from typing import TypeAlias

    # Defined here rather than imported from the private `polars._typing`.
    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series
    ClosedInterval: TypeAlias = Literal["left", "right", "both", "none"]

__all__ = ["__version__", "rolling_xsum", "rolling_xsum_by", "thread_pool_size", "xsum"]

_PLUGIN_PATH = Path(__file__).parent


def xsum(expr: IntoExprColumn) -> pl.Expr:
    """Return the exact sum of a float column, correctly rounded.

    The values are summed *without any rounding error*, and only the final
    result is rounded (to nearest, ties to even). The result therefore does not
    depend on the order of the values, on how the column is chunked, or on the
    number of threads, and cancellation cannot destroy precision.

    Parameters
    ----------
    expr
        A ``Float32`` or ``Float64`` expression, a column name, or a Series. Other
        dtypes, including integers and ``Float16``, are rejected with a
        :class:`polars.exceptions.ComputeError`. To opt in, cast explicitly
        (e.g. ``.cast(pl.Float64)``). Keep in mind that integers beyond 2**53
        are not exactly representable as ``Float64``.

    Returns
    -------
    pl.Expr
        A scalar expression of the input's dtype. It works in ``select``,
        ``with_columns``, ``group_by(...).agg(...)`` and ``over(...)``.

    Notes
    -----
    * Nulls are skipped. An empty or all-null column sums to ``0.0``, as with
      :meth:`polars.Expr.sum`.
    * Any NaN gives NaN. Infinities of both signs give NaN. Otherwise an
      infinite value gives that infinity.
    * An exact sum beyond the dtype's finite range gives ``±inf``.
      Intermediate overflow does not, e.g. ``[MAX, MAX, -MAX]`` sums to
      ``MAX``.
    * An exact zero is returned as ``+0.0``.
    * ``Float32`` sums are rounded directly to ``Float32``, with no double
      rounding through ``Float64``.
    * Columns of at least 32Ki values are summed in parallel on the plugin's
      own thread pool, whose size follows ``POLARS_MAX_THREADS`` (see
      :func:`thread_pool_size`). The result is the same for any number of
      threads.
    * In ``group_by(...).agg(...)`` and ``over(...)``, all groups are summed
      in a single plugin call, in parallel. Polars first gathers the values
      of each group, as it does for any non-built-in aggregation. With
      overlapping windows (``rolling``, ``group_by_dynamic``), every window is
      gathered, so memory grows with the number of rows times the window
      length. For one sum per row, use :func:`rolling_xsum` or
      :func:`rolling_xsum_by` instead of ``rolling``.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_xsum import xsum
    >>> df = pl.DataFrame({"x": [1e100, 1.0, -1e100], "g": ["a", "a", "b"]})
    >>> df.select(naive=pl.col("x").sum(), exact=xsum("x"))
    shape: (1, 2)
    ┌───────┬───────┐
    │ naive ┆ exact │
    │ ---   ┆ ---   │
    │ f64   ┆ f64   │
    ╞═══════╪═══════╡
    │ 0.0   ┆ 1.0   │
    └───────┴───────┘
    >>> df.group_by("g", maintain_order=True).agg(xsum("x"))
    shape: (2, 2)
    ┌─────┬─────────────┐
    │ g   ┆ x           │
    │ --- ┆ ---         │
    │ str ┆ f64         │
    ╞═════╪═════════════╡
    │ a   ┆ 1.0000e100  │
    │ b   ┆ -1.0000e100 │
    └─────┴─────────────┘
    """
    if isinstance(expr, str):
        expr = pl.col(expr)
    elif isinstance(expr, pl.Series):
        expr = pl.lit(expr)
    # Imploding gives one list per group in `group_by` and `over` (and a single
    # list otherwise). The elementwise list sum then sums all groups in one
    # plugin call, instead of one call per group.
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="xsum_list",
        args=expr.implode(),
        is_elementwise=True,
    )


def rolling_xsum(
    expr: IntoExprColumn,
    window_size: int,
    *,
    min_samples: int | None = None,
    center: bool = False,
) -> pl.Expr:
    """Return the exact sum of every window of ``window_size`` rows.

    This is the exact counterpart of :meth:`polars.Expr.rolling_sum` (without
    ``weights``). Each window sum is exact and rounded once, with the rules of
    :func:`xsum`, so it does not depend on the values outside the window.

    Parameters
    ----------
    expr
        A ``Float32`` or ``Float64`` expression, a column name, or a Series.
    window_size
        Number of rows per window. The window of a row ends at that row, or is
        centred on it with ``center=True``.
    min_samples
        Number of non-null values a window needs for a non-null result.
        Defaults to ``window_size``.
    center
        Centre the window on its row instead of ending it there.

    Returns
    -------
    pl.Expr
        An expression of the input's dtype and length.

    Notes
    -----
    * Nulls are skipped. With ``min_samples=0``, a window without values
      sums to ``0.0``.
    * The window slides over the column with a single exact accumulator: the
      value entering the window is added, the value leaving it subtracted.
      The cost per row is therefore independent of the window size. Long
      columns are processed in parallel on the plugin's thread pool.
    * In ``over(...)``, the windows stay within each group. The plugin is
      then called once per group, so with many small groups the per-call
      overhead dominates.

    Examples
    --------
    >>> import polars as pl
    >>> from polars_xsum import rolling_xsum
    >>> df = pl.DataFrame({"x": [1e100, 1.0, -1e100]})
    >>> df.with_columns(
    ...     naive=pl.col("x").rolling_sum(3, min_samples=1),
    ...     exact=rolling_xsum("x", 3, min_samples=1),
    ... )
    shape: (3, 3)
    ┌─────────────┬────────────┬────────────┐
    │ x           ┆ naive      ┆ exact      │
    │ ---         ┆ ---        ┆ ---        │
    │ f64         ┆ f64        ┆ f64        │
    ╞═════════════╪════════════╪════════════╡
    │ 1.0000e100  ┆ 1.0000e100 ┆ 1.0000e100 │
    │ 1.0         ┆ 1.0000e100 ┆ 1.0000e100 │
    │ -1.0000e100 ┆ 0.0        ┆ 1.0        │
    └─────────────┴────────────┴────────────┘
    """
    _check_count("window_size", window_size)
    if min_samples is None:
        min_samples = window_size
    _check_count("min_samples", min_samples)
    if isinstance(expr, str):
        expr = pl.col(expr)
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="rolling_xsum",
        args=expr,
        kwargs={"window_size": window_size, "min_samples": min_samples, "center": center},
        is_elementwise=False,
    )


def rolling_xsum_by(
    expr: IntoExprColumn,
    by: IntoExprColumn,
    window_size: timedelta | str,
    *,
    min_samples: int = 0,
    closed: ClosedInterval = "right",
) -> pl.Expr:
    """Return the exact sum of every window of a key column.

    This is the exact counterpart of :meth:`polars.Expr.rolling_sum_by`. The
    window of a row holds the rows whose key lies in ``(key - window_size,
    key]`` (by default), and its sum is exact and rounded once, with the rules
    of :func:`xsum`.

    Parameters
    ----------
    expr
        A ``Float32`` or ``Float64`` expression, a column name, or a Series.
    by
        The key column: an integer, ``Date`` or ``Datetime`` column. It need
        not be sorted.
    window_size
        The length of the windows: a number of key units such as ``"3i"`` for
        an integer key, otherwise a duration such as ``"2d"``, ``"1mo"`` or
        ``"1d12h"`` (the language of :meth:`polars.Expr.dt.offset_by`, so
        calendar units and time zones are respected), or a ``timedelta``.
    min_samples
        Number of non-null values a window needs for a non-null result. The
        default, 0, is that of Polars 2 (Polars 1 defaults to 1).
    closed
        Which ends of the window interval are included: ``"right"``
        (default), ``"left"``, ``"both"`` or ``"none"``.

    Returns
    -------
    pl.Expr
        An expression of the input's dtype and length.

    Notes
    -----
    * Nulls are skipped. A row with a null key gives null and belongs to no
      window.
    * Rows are sorted by key if necessary. The window then slides over them
      with a single exact accumulator, adding values that enter it and
      subtracting values that leave it, so the cost per row does not depend
      on the window size.
    * In ``over(...)``, the windows stay within each group. The plugin is
      then called once per group, so with many small groups the per-call
      overhead dominates.

    Examples
    --------
    >>> from datetime import date
    >>> import polars as pl
    >>> from polars_xsum import rolling_xsum_by
    >>> df = pl.DataFrame(
    ...     {
    ...         "day": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 4)],
    ...         "x": [1e100, 1.0, -1e100],
    ...     }
    ... )
    >>> df.with_columns(
    ...     naive=pl.col("x").rolling_sum_by("day", "4d"),
    ...     exact=rolling_xsum_by("x", "day", "4d"),
    ... )
    shape: (3, 4)
    ┌────────────┬─────────────┬────────────┬────────────┐
    │ day        ┆ x           ┆ naive      ┆ exact      │
    │ ---        ┆ ---         ┆ ---        ┆ ---        │
    │ date       ┆ f64         ┆ f64        ┆ f64        │
    ╞════════════╪═════════════╪════════════╪════════════╡
    │ 2024-01-01 ┆ 1.0000e100  ┆ 1.0000e100 ┆ 1.0000e100 │
    │ 2024-01-02 ┆ 1.0         ┆ 1.0000e100 ┆ 1.0000e100 │
    │ 2024-01-04 ┆ -1.0000e100 ┆ 0.0        ┆ 1.0        │
    └────────────┴─────────────┴────────────┴────────────┘
    """
    if isinstance(window_size, timedelta):
        window_size = _duration_string(window_size)
    elif not isinstance(window_size, str):
        msg = f"window_size must be a str or timedelta, got {type(window_size).__name__}"
        raise TypeError(msg)
    if window_size.startswith("-"):
        msg = f"window_size must be positive, got {window_size!r}"
        raise ValueError(msg)
    _check_count("min_samples", min_samples)
    if closed not in ("left", "right", "both", "none"):
        msg = f"closed must be 'left', 'right', 'both' or 'none', got {closed!r}"
        raise ValueError(msg)
    if isinstance(expr, str):
        expr = pl.col(expr)
    by = pl.col(by) if isinstance(by, str) else pl.lit(by) if isinstance(by, pl.Series) else by
    if re.fullmatch(r"\d+i", window_size):
        # The plugin computes `key - size` itself, without overflow.
        size = int(window_size[:-1])
        _check_count("window_size", size)
        if size == 0:
            msg = f"window_size must be positive, got {window_size!r}"
            raise ValueError(msg)
        args = [expr, by]
    else:
        # Polars computes the lower end of each row's window, so that calendar
        # durations and time zones follow `dt.offset_by`.
        size = None
        args = [expr, by, by.dt.offset_by(f"-{window_size}")]
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="rolling_xsum_by",
        args=args,
        kwargs={"min_samples": min_samples, "closed": closed, "window_size": size},
        is_elementwise=False,
    )


def _check_count(name: str, value: object) -> None:
    """Raise unless ``value`` is an integer in ``[0, 2**63)``."""
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"{name} must be an int, got {type(value).__name__}"
        raise TypeError(msg)
    if not 0 <= value < 2**63:
        msg = f"{name} must be non-negative and below 2**63, got {value}"
        raise ValueError(msg)


def _duration_string(delta: timedelta) -> str:
    """Return ``delta`` in Polars' duration language, e.g. ``"1d3600s"``."""
    if delta <= timedelta(0):
        msg = f"window_size must be positive, got {delta}"
        raise ValueError(msg)
    parts = ((delta.days, "d"), (delta.seconds, "s"), (delta.microseconds, "us"))
    return "".join(f"{value}{unit}" for value, unit in parts if value)
