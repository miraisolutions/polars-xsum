//! The plugin functions behind the `xsum`, `rolling_xsum` and `rolling_xsum_by`
//! expressions, as seen by Polars.
//!
//! The `#[polars_expr]` macro generates the `unsafe extern "C"` entry points of
//! the Polars plugin ABI. Those entry points are the only unsafe code in this
//! crate, so this module is the only place where `unsafe_code` is allowed.
//!
//! The macro moves each annotated function into its entry points, so rustdoc
//! shows only the entry points, without the functions' documentation:
//!
//! | Expression        | Entry points                                                             | Computes                                      |
//! |-------------------|--------------------------------------------------------------------------|-----------------------------------------------|
//! | `xsum`            | `_polars_plugin_xsum_list`, `_polars_plugin_field_xsum_list`             | the exact sum of each list of a list column   |
//! | `rolling_xsum`    | `_polars_plugin_rolling_xsum`, `_polars_plugin_field_rolling_xsum`       | the exact sum of every window of rows         |
//! | `rolling_xsum_by` | `_polars_plugin_rolling_xsum_by`, `_polars_plugin_field_rolling_xsum_by` | the exact sum of every window of a key column |
//!
//! The `_field_` entry points compute the output schema, with the
//! `*_output_type` functions below.
#![allow(
    unsafe_code,
    reason = "#[polars_expr] generates the unsafe extern \"C\" plugin entry points"
)]

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use xsum_core::SmallAccumulator;

use crate::column::exact_list_sums;
use crate::rolling::{KeyWindow, Window, exact_rolling_sums, exact_rolling_sums_by};

/// Output schema of `xsum_list`: the float type of the list elements.
///
/// The input is the user's column imploded into lists, so errors report the
/// element type, i.e. the dtype of the user's column.
///
/// # Errors
///
/// Fails if the input is not a list column, or if its elements are not
/// `Float32` or `Float64`.
fn xsum_list_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => float_field("xsum", field, inner),
        dtype => polars_bail!(InvalidOperation: "xsum_list requires a List column, got {dtype}"),
    }
}

/// Returns a field with the name of `field` and the dtype `dtype`.
///
/// # Errors
///
/// Fails if `dtype` is not `Float32` or `Float64`. The message names the
/// plugin function `function` and suggests an explicit cast.
fn float_field(function: &str, field: &Field, dtype: &DataType) -> PolarsResult<Field> {
    match dtype {
        DataType::Float32 | DataType::Float64 => {
            Ok(Field::new(field.name().clone(), dtype.clone()))
        }
        dtype => polars_bail!(
            InvalidOperation:
            "{function} requires a Float32 or Float64 column, but '{}' has dtype {dtype}; \
             cast it explicitly if a conversion is intended, e.g. `pl.col(\"{}\").cast(pl.Float64)`",
            field.name(),
            field.name()
        ),
    }
}

/// Exact, correctly rounded sum of each list of a `List(Float32|Float64)`
/// column. Nulls inside the lists are skipped; a null list gives null.
///
/// `xsum` applies this to its imploded input, so that Polars calls the plugin
/// once for all groups of a `group_by` or `over`, not once per group.
///
/// # Errors
///
/// Fails if the input is not a list column of `Float32` or `Float64` values
/// (which [`xsum_list_output_type`] normally rejects earlier).
#[polars_expr(output_type_func = xsum_list_output_type)]
fn xsum_list(inputs: &[Series]) -> PolarsResult<Series> {
    let lists = inputs[0].list()?;
    match lists.inner_dtype() {
        DataType::Float32 => {
            Ok(exact_list_sums::<Float32Type>(lists, SmallAccumulator::to_f32)?.into_series())
        }
        DataType::Float64 => {
            Ok(exact_list_sums::<Float64Type>(lists, SmallAccumulator::to_f64)?.into_series())
        }
        dtype => {
            polars_bail!(InvalidOperation: "xsum requires a Float32 or Float64 column, got {dtype}")
        }
    }
}

/// Output schema of `rolling_xsum`: the input float type.
///
/// # Errors
///
/// Fails if the input is not a `Float32` or `Float64` column.
fn rolling_xsum_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    float_field("rolling_xsum", &input_fields[0], input_fields[0].dtype())
}

/// Exact, correctly rounded sum of every window of the input column, with the
/// window given by `kwargs`. Nulls are skipped; a window with fewer than
/// `min_samples` non-null values gives null.
///
/// # Errors
///
/// Fails if the input is not a `Float32` or `Float64` column, or if
/// `min_samples` exceeds `window_size`.
#[polars_expr(output_type_func = rolling_xsum_output_type)]
fn rolling_xsum(inputs: &[Series], kwargs: Window) -> PolarsResult<Series> {
    let series = &inputs[0];
    match series.dtype() {
        DataType::Float32 => {
            Ok(exact_rolling_sums(series.f32()?, kwargs, SmallAccumulator::to_f32)?.into_series())
        }
        DataType::Float64 => {
            Ok(exact_rolling_sums(series.f64()?, kwargs, SmallAccumulator::to_f64)?.into_series())
        }
        dtype => {
            polars_bail!(InvalidOperation: "rolling_xsum requires a Float32 or Float64 column, got {dtype}")
        }
    }
}

/// Output schema of `rolling_xsum_by`: the float type of the values.
///
/// # Errors
///
/// Fails if the key column is not an integer, `Date` or `Datetime` column, or
/// if the values are not `Float32` or `Float64`.
fn rolling_xsum_by_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let by = &input_fields[1];
    polars_ensure!(
        by.dtype().is_integer() || matches!(by.dtype(), DataType::Date | DataType::Datetime(..)),
        InvalidOperation: "rolling_xsum_by requires an integer, Date or Datetime key column, \
                           but '{}' has dtype {}",
        by.name(),
        by.dtype()
    );
    float_field("rolling_xsum_by", &input_fields[0], input_fields[0].dtype())
}

/// Exact, correctly rounded sum of every window of a key column.
///
/// # Arguments
///
/// * `inputs` - The values and the key column, plus, for a duration window,
///   the lower end of each row's window (computed by Polars with
///   `dt.offset_by`).
/// * `kwargs` - The remaining window parameters (see `KeyWindow`).
///
/// # Errors
///
/// Fails if there are not 2 or 3 inputs, if the values are not `Float32` or
/// `Float64`, if an integer window size such as `"3i"` is used with a
/// temporal key column, if a key or lower end does not fit into an `Int64`
/// (an unsigned key above `i64::MAX`), or if the columns differ in length.
#[polars_expr(output_type_func = rolling_xsum_by_output_type)]
fn rolling_xsum_by(inputs: &[Series], kwargs: KeyWindow) -> PolarsResult<Series> {
    let (values, by, lower) = match inputs {
        [values, by] => (values, by, None),
        [values, by, lower] => (values, by, Some(lower)),
        _ => {
            polars_bail!(ComputeError: "rolling_xsum_by expects 2 or 3 inputs, got {}", inputs.len())
        }
    };
    // A duration window on a non-temporal key fails earlier, in Polars'
    // `dt.offset_by`.
    polars_ensure!(
        kwargs.window_size.is_none() || by.dtype().is_integer(),
        InvalidOperation: "a window size like '3i' requires an integer key column, \
                           but '{}' has dtype {}; use a duration such as '3d'",
        by.name(),
        by.dtype()
    );
    let physical = |s: &Series| s.to_physical_repr().strict_cast(&DataType::Int64);
    let keys = physical(by)?;
    let lower = lower.map(physical).transpose()?;
    let (keys, lower) = (keys.i64()?, lower.as_ref().map(Series::i64).transpose()?);
    match values.dtype() {
        DataType::Float32 => Ok(exact_rolling_sums_by(
            values.f32()?,
            keys,
            lower,
            kwargs,
            SmallAccumulator::to_f32,
        )?
        .into_series()),
        DataType::Float64 => Ok(exact_rolling_sums_by(
            values.f64()?,
            keys,
            lower,
            kwargs,
            SmallAccumulator::to_f64,
        )?
        .into_series()),
        dtype => {
            polars_bail!(InvalidOperation: "rolling_xsum_by requires a Float32 or Float64 column, got {dtype}")
        }
    }
}
