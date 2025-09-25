use crate::util::binary_amortized_elementwise;
use polars::{prelude::*, series::amortized_iter::AmortSeries};
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct SnippetMeanKwargs {
    aggregation: String,
    x_min: f64,
    x_max: f64,
    x_range_excluded: Option<(f64, f64)>,
    x_min_idx_offset: Option<usize>,
    x_max_idx_offset: Option<usize>,
}

/// Compute the aggregation of a range of elements of a `List` column, where the
/// range is defined by the values in another `List` column and the aggregation
/// method is one of "mean", "median", "sum", "count", "max", or "min".
///
/// The range is inclusive of the `x_min` and `x_max` values.
///
/// ## Parameters
/// - `aggregation`: The aggregation method to use.
/// - `list_column_y`: The `List` column of samples to compute the aggregation of.
/// - `list_column_x`: The `List` column of samples to use as the range.
/// - `x_min`: The minimum value of the range.
/// - `x_max`: The maximum value of the range.
/// - `x_range_excluded`: Optional range of values to exclude from the range.
/// - `x_min_idx_offset`: The index offset to add to the `x_min` constraint.
/// - `x_max_idx_offset`: The index offset to subtract from the `x_max` constraint.
///
/// ## Return value
/// New `Float64` column with the aggregation of the elements in the range.
#[polars_expr(output_type=Float64)]
fn expr_agg_of_range(
    inputs: &[Series],
    kwargs: SnippetMeanKwargs,
) -> PolarsResult<Series> {
    let input_y = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let input_x = inputs[1].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let y = input_y.list()?;
    let x = input_x.list()?;

    let valid_aggregations = ["mean", "median", "sum", "count", "max", "min"];
    if !valid_aggregations.contains(&kwargs.aggregation.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "(agg_of_range): Invalid aggregation method provided: {}. Must be one of [{}]",
                kwargs.aggregation,
                valid_aggregations.join(", "),
            )
            .into(),
        ));
    }

    if let Some((min_excl, max_excl)) = kwargs.x_range_excluded
        && (min_excl.is_nan() || max_excl.is_nan())
    {
        return Err(PolarsError::ComputeError(
            "(agg_of_range): x_range_excluded values must be valid numbers (not NaN)"
                .into(),
        ));
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
                if let (Some(y), Some(x)) = (y, x)
                    && !x.is_nan()
                    && !y.is_nan()
                    && (kwargs.x_min..=kwargs.x_max).contains(&x)
                {
                    if let Some((min_excl, max_excl)) = kwargs.x_range_excluded
                        && (min_excl..=max_excl).contains(&x)
                    {
                        return;
                    }
                    acc_values.push(y);
                    counter += 1;
                }
            });

            let upp_limit = acc_values.len() - kwargs.x_max_idx_offset.unwrap_or(0);
            let low_limit = kwargs.x_min_idx_offset.unwrap_or(0);
            let mut acc_values: Vec<f64> = acc_values
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
                match kwargs.aggregation.as_str() {
                    "mean" => {
                        Some(acc_values.iter().sum::<f64>() / acc_values.len() as f64)
                    }
                    "median" => {
                        acc_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        let mid = acc_values.len() / 2;
                        if acc_values.len().is_multiple_of(2) {
                            Some((acc_values[mid - 1] + acc_values[mid]) / 2.0)
                        } else {
                            Some(acc_values[mid])
                        }
                    }
                    "sum" => Some(acc_values.iter().sum()),
                    "count" => Some(acc_values.len() as f64),
                    "max" => acc_values.into_iter().reduce(f64::max),
                    "min" => acc_values.into_iter().reduce(f64::min),
                    _ => unreachable!(),
                }
            }
        },
    );

    Ok(out.into_series())
}
