use crate::util::same_dtype;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct AggregateListKwargs {
    list_size: usize,
    aggregation: String,
}

/// Aggregate the elements, column-wise, of a `List[f64]` column.
///
/// The function raises an Error if:
/// * the samples column is not of type `List(Float64)`
/// * the aggregation method is not one of "mean", "sum", or "count"
/// * the list_size is 0
/// * any of the lists in the column is shorter than list_size
///
/// ## Parameters
/// - `list_size`: The size of each list in the `List[f64]` column.
/// - `aggregation`: The aggregation method to use. One of "mean", "sum", or "count".
///
/// ## Return value
/// New `List[f64]` column with the result of the aggregation.
#[polars_expr(output_type_func=same_dtype)]
fn expr_aggregate_list_col_elementwise(
    inputs: &[Series],
    kwargs: AggregateListKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    if ca.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", ca.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    if kwargs.list_size == 0 {
        return Err(PolarsError::ComputeError(
            "list_size must be greater than 0".into(),
        ));
    }

    let valid_aggregations = ["mean", "sum", "count"];
    if !valid_aggregations.contains(&kwargs.aggregation.as_str()) {
        return Err(PolarsError::ComputeError(
            format!(
                "Invalid aggregation method provided: {}. Must be one of [{}]",
                kwargs.aggregation,
                valid_aggregations.join(", "),
            )
            .into(),
        ));
    }

    let mut buckets: Vec<Option<f64>> = vec![None; kwargs.list_size];
    let mut counts: Vec<usize> = vec![0; kwargs.list_size];

    let mut list_too_short = false;
    let dummy_vec: Vec<f64> = Vec::new();

    // Collect the `List[f64]` values into a Vec<f64>
    let _ = ca.apply_amortized(|s| {
        let s: &Series = s.as_ref();
        let ca: &Float64Chunked = s.f64().unwrap();

        // Collect the `List[f64]` values into a Vec<f64>
        let elements: Vec<Option<f64>> = ca
            .iter()
            .zip(buckets.iter_mut())
            .zip(counts.iter_mut())
            .map(|((val, bucket), count)| {
                if !val.is_none_or(|x| x.is_nan() || x.is_infinite()) {
                    if let Some(bucket) = bucket {
                        *bucket += val.unwrap();
                    } else {
                        *bucket = Some(val.unwrap());
                    }
                    *count += 1;
                }
                val
            })
            .collect();

        // Check if any of the lists is shorter than the given list_size
        if elements.len() < kwargs.list_size {
            list_too_short = true;
        }

        // Return as Series
        Series::new(PlSmallStr::EMPTY, dummy_vec.clone())
    });

    let out = match kwargs.aggregation.as_str() {
        "mean" => buckets
            .iter()
            .zip(counts.iter())
            .map(|(bucket, count)| {
                if *count == 0 || bucket.is_none() {
                    None
                } else {
                    Some(bucket.unwrap() / *count as f64)
                }
            })
            .collect::<Vec<Option<f64>>>(),

        "sum" => buckets,

        "count" => counts
            .iter()
            .map(|count| Some(*count as f64))
            .collect::<Vec<Option<f64>>>(),

        // We already checked for this
        _ => unreachable!(),
    };

    if list_too_short {
        Err(PolarsError::ComputeError(
            "One of the lists is shorter than the given list_size!".into(),
        ))
    } else {
        Ok(Series::new(PlSmallStr::EMPTY, out))
    }
}
