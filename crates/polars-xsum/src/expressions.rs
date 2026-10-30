//! The plugin functions behind the `xsum`, `rolling_xsum` and `rolling_xsum_by`
//! expressions, as seen by Polars.
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
use xsum_core::SmallAccumulator;

use crate::column::exact_list_sums;
use crate::rolling::{KeyWindow, Window, exact_rolling_sums, exact_rolling_sums_by};

/// Output schema of `xsum_list`: the float type of the list elements.
///
/// The input is the user's column imploded into lists, so errors report the
/// element type, i.e. the dtype of the user's column.
fn xsum_list_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::List(inner) => float_field("xsum", field, inner),
        dtype => polars_bail!(InvalidOperation: "xsum_list requires a List column, got {dtype}"),
    }
}

/// `field` renamed to `dtype` if that is Float32 or Float64; an error naming
/// `function` otherwise.
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
fn rolling_xsum_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    float_field("rolling_xsum", &input_fields[0], input_fields[0].dtype())
}

/// Exact, correctly rounded sum of every window of `window_size` rows. Nulls
/// are skipped; a window with fewer than `min_samples` non-null values gives
/// null.
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

/// Output schema of `rolling_xsum_by`: the float type of the values. The key
/// column must be an integer, `Date` or `Datetime` column.
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

/// Exact, correctly rounded sum of every window of a key column. The inputs
/// are the values and the key column, plus, for a duration window, the lower
/// end of each row's window (computed by Polars, see `KeyWindow`).
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
