//! Polars expression plugin exposing exact floating-point summation.
//!
//! The Python package `polars_xsum` registers the expressions `xsum`,
//! `rolling_xsum` and `rolling_xsum_by`, whose plugin functions are defined in
//! the `expressions` module. All numerics live in the `xsum-core` crate. This
//! crate maps Polars columns, which may be chunked and may contain nulls, onto
//! the superaccumulators, slides them over rolling windows, and runs parallel
//! work on its own thread pool (`pool`).

mod column;
mod expressions;
mod pool;
mod rolling;

use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3_polars::PolarsAllocator;

/// Share Polars' allocator so memory can be freed by either side of the
/// plugin boundary.
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

// pyo3 turns this doc comment into the Python docstring of
// `polars_xsum.thread_pool_size`, which Sphinx renders in the API reference;
// hence the reStructuredText markup.
/// Returns the number of threads of the plugin's thread pool.
///
/// ``xsum``, ``rolling_xsum`` and ``rolling_xsum_by`` use this pool for long
/// inputs.
///
/// The plugin has its own thread pool (it cannot share the Polars host's). Its
/// size follows ``POLARS_MAX_THREADS`` and defaults to the number of available
/// CPUs; ``RAYON_NUM_THREADS`` overrides it. Both are read once, when the pool
/// is first used.
#[pyfunction]
fn thread_pool_size() -> usize {
    pool::size()
}

/// The compiled extension module `polars_xsum._internal`.
///
/// # Errors
///
/// Fails if `__version__` or `thread_pool_size` cannot be added to the module.
#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(thread_pool_size, m)?)?;
    Ok(())
}
