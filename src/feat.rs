use crate::util::binary_amortized_elementwise;
use polars::{prelude::*, series::amortized_iter::AmortSeries};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct SnippetMeanKwargs {
    x_min: f64,
    x_max: f64,
    x_min_idx_offset: Option<usize>,
    x_max_idx_offset: Option<usize>,
}

/// Compute the mean of a range of elements of a `List[f64]` column, where the
/// range is defined by the values in another `List[f64]` column.
///
/// The range is inclusive of the `x_min` and `x_max` values.
///
/// The function raises an Error if:
/// * the y column is not of type `List(Float64)`
/// * the x column is not of type `List(Float64)`
///
/// ## Parameters
/// - `list_column_y`: The `List[f64]` column of samples to compute the mean of.
/// - `list_column_x`: The `List[f64]` column of samples to use as the range.
/// - `x_min`: The minimum value of the range.
/// - `x_max`: The maximum value of the range.
/// - `x_min_idx_offset`: The index offset to add to the `x_min` constraint.
/// - `x_max_idx_offset`: The index offset to subtract from the `x_max` constraint.
///
/// ## Return value
/// New `Float64` column with the mean of the elements in the range.
#[polars_expr(output_type=Float64)]
fn expr_mean_of_range(
    inputs: &[Series],
    kwargs: SnippetMeanKwargs,
) -> PolarsResult<Series> {
    let y = inputs[0].list()?;
    let x = &inputs[1].list()?;

    if y.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", y.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    if x.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", x.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    let out: Float64Chunked = binary_amortized_elementwise(
        y,
        x,
        |y_inner: &AmortSeries, x_inner: &AmortSeries| -> Option<f64> {
            let y_inner = y_inner.as_ref().f64().unwrap();
            let x_inner = x_inner.as_ref().f64().unwrap();

            let mut acc_values: Vec<f64> = Vec::with_capacity(y_inner.len());
            let mut counter: usize = 0;

            y_inner.iter().zip(x_inner.iter()).for_each(|(y, x)| {
                if let (Some(y), Some(x)) = (y, x) {
                    if !x.is_nan()
                        && !y.is_nan()
                        && (kwargs.x_min..=kwargs.x_max).contains(&x)
                    {
                        acc_values.push(y);
                        counter += 1;
                    }
                }
            });

            let upp_limit = acc_values.len() - kwargs.x_max_idx_offset.unwrap_or(0);
            let low_limit = kwargs.x_min_idx_offset.unwrap_or(0);
            let acc_values: Vec<f64> = acc_values
                .into_iter()
                .enumerate()
                .filter_map(|(idx, y)| {
                    if idx >= low_limit && idx < upp_limit {
                        Some(y)
                    } else {
                        None
                    }
                })
                .collect();

            if acc_values.is_empty() {
                None
            } else {
                Some(acc_values.iter().sum::<f64>() / acc_values.len() as f64)
            }
        },
    );

    Ok(out.into_series())
}
