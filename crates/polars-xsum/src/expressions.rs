//! The `xsum` expression as seen by Polars.
//!
//! The `#[polars_expr]` macro generates the `unsafe extern "C"` entry points of
//! the Polars plugin ABI. Those entry points are the only unsafe code in this
//! crate, so this module is the only place where `unsafe_code` is allowed.
#![allow(
    unsafe_code,
    reason = "#[polars_expr] generates the unsafe extern \"C\" plugin entry points"
)]

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

use crate::column::exact_sum;

/// Output schema of `xsum`: the input float type. Any other type is rejected.
fn xsum_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::Float32 | DataType::Float64 => Ok(field.clone()),
        dtype => polars_bail!(
            InvalidOperation:
            "xsum requires a Float32 or Float64 column, but '{}' has dtype {dtype}; \
             cast it explicitly if a conversion is intended, e.g. `pl.col(\"{}\").cast(pl.Float64)`",
            field.name(),
            field.name()
        ),
    }
}

/// Exact, correctly rounded sum of a float column. Nulls are skipped.
#[polars_expr(output_type_func = xsum_output_type)]
fn xsum(inputs: &[Series]) -> PolarsResult<Series> {
    let series = &inputs[0];
    let name = series.name().clone();
    match series.dtype() {
        DataType::Float32 => Ok(Series::new(name, [exact_sum(series.f32()?).to_f32()])),
        DataType::Float64 => Ok(Series::new(name, [exact_sum(series.f64()?).to_f64()])),
        dtype => {
            polars_bail!(InvalidOperation: "xsum requires a Float32 or Float64 column, got {dtype}")
        }
    }
}
