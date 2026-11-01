# API reference

```{eval-rst}
.. automodule:: polars_xsum
   :no-members:

.. autofunction:: polars_xsum.xsum

.. autofunction:: polars_xsum.rolling_xsum

.. autofunction:: polars_xsum.rolling_xsum_by

.. autofunction:: polars_xsum.thread_pool_size

.. py:data:: polars_xsum.__version__
   :type: str

   The version of the installed package.

Type aliases
------------

These aliases appear in the signatures above. They exist for type checkers
only and cannot be imported at runtime.

.. py:type:: polars_xsum.IntoExprColumn
   :canonical: polars.Expr | str | polars.Series

   An expression, the name of a column, or a :class:`polars.Series`.

.. py:type:: polars_xsum.ClosedInterval
   :canonical: Literal["left", "right", "both", "none"]

   Which ends of a window interval are included.
```
