use crate::util::list_f64_dtype;
use interp::{InterpMode, interp_slice};
use polars::prelude::*;
use pyo3_polars::{
    derive::polars_expr, export::polars_core::utils::align_chunks_ternary,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct OperateScalarListKwargs {
    operation: String,
}

/// Apply a scalar operation elementwise to a `List` column.
///
/// The function raises an Error if:
/// * the operation is not one of "add", "sub", "mul" or "div"
///
/// ## Parameters
/// - `list_columns`: The `List` column to apply the operation to.
/// - `scalar_column`: The `Float64` scalar to apply the operation with.
/// - `operation`: The operation to apply. Must be one of "add", "sub", "mul" or "div".
///
/// ## Return value
/// New `List[f64]` column with the result of the operation.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_operate_scalar_on_list(
    inputs: &[Series],
    kwargs: OperateScalarListKwargs,
) -> PolarsResult<Series> {
    let input_list = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_scalar = inputs[1].cast(&DataType::Float64)?;
    let list = input_list.list()?;
    let scalar = input_scalar.f64()?;

    let valid_operations = ["div", "mul", "add", "sub"];
    if !valid_operations.contains(&kwargs.operation.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(operate_scalar_on_list): Invalid operation method provided: {}. Must be one of [{}]",
                kwargs.operation,
                valid_operations.join(", "),
            )
            .into(),
        ));
    }

    let out: ListChunked = list.zip_and_apply_amortized(scalar, |ca_list, ca_scalar| {
        if let (Some(ca_list), Some(ca_scalar)) = (ca_list, ca_scalar) {
            let list: Vec<Option<f64>> = ca_list
                .as_ref()
                .f64()
                .unwrap()
                .iter()
                .map(|val| {
                    if let Some(val) = val {
                        Some(match kwargs.operation.as_str() {
                            "div" => val / ca_scalar,
                            "mul" => val * ca_scalar,
                            "add" => val + ca_scalar,
                            "sub" => val - ca_scalar,
                            _ => unreachable!(),
                        })
                    } else {
                        None
                    }
                })
                .collect();

            Some(Series::new(PlSmallStr::EMPTY, list))
        } else {
            None
        }
    });

    Ok(out.into_series())
}

/// Interpolate columns to obtain `y_interp` from `x_data`, `x_interp` and `y_data`.
///
/// ## Parameters
/// - `x_data`: The `List` column containing the x-coords of the data.
/// - `y_data`: The `List` column containing the y-coords of the data.
/// - `x_interp`: The `List` column containing the x-coords of the interpolation.
///
/// ## Return value
/// New `List[f64]` column with the new y-coords of the interpolation, `y_interp`.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_interpolate_columns(inputs: &[Series]) -> PolarsResult<Series> {
    let input_x_data = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_y_data = inputs[1].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_x_interp = inputs[2].cast(&DataType::List(Box::new(DataType::Float64)))?;

    let x_data = input_x_data.list()?;
    let y_data = input_y_data.list()?;
    let x_interp = input_x_interp.list()?;

    let (x_data, y_data, x_interp) = align_chunks_ternary(x_data, y_data, x_interp);

    let out: ListChunked = x_data
        .amortized_iter()
        .zip(y_data.amortized_iter())
        .zip(x_interp.amortized_iter())
        .map(|((x, y), x_interp)| {
            if let (Some(x), Some(y), Some(x_interp)) = (x, y, x_interp) {
                let x: Vec<f64> = x
                    .as_ref()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|val| val.unwrap_or(f64::NAN))
                    .collect();

                let y: Vec<f64> = y
                    .as_ref()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|val| val.unwrap_or(f64::NAN))
                    .collect();

                let x_interp: Vec<f64> = x_interp
                    .as_ref()
                    .f64()
                    .unwrap()
                    .iter()
                    .map(|val| val.unwrap_or(f64::NAN))
                    .collect();

                let interpolated =
                    interp_slice(&x, &y, &x_interp, &InterpMode::FirstLast);

                Some(Series::new(PlSmallStr::EMPTY, interpolated))
            } else {
                None
            }
        })
        .collect();

    Ok(out.into_series())
}
