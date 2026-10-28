"""Exact, correctly rounded summation of floating-point columns for Polars.

The :func:`xsum` expression computes the *exact* sum of a ``Float32`` or
``Float64`` column and rounds it once to the column's type. It uses Radford M.
Neal's small and large superaccumulators (arXiv:1505.05571) and is implemented
in Rust.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

from polars_xsum._internal import __version__, thread_pool_size

if TYPE_CHECKING:
    from typing import TypeAlias

    # Defined here rather than imported from the private `polars._typing`.
    IntoExprColumn: TypeAlias = pl.Expr | str | pl.Series

__all__ = ["__version__", "thread_pool_size", "xsum"]

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
    return register_plugin_function(
        plugin_path=_PLUGIN_PATH,
        function_name="xsum",
        args=expr,
        is_elementwise=False,
        returns_scalar=True,
    )
