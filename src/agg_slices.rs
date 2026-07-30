use crate::slice::{Range, Slice};
use crate::statistic::Statistic;
use crate::util::ListInputs;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum Aggregation {
    Mean,
    Median,
    Min,
    Max,
    Std,
    Delta,
    Count,
}

impl Aggregation {
    fn apply(
        &self,
        values: &[f64],
    ) -> Option<f64> {
        match self {
            Self::Mean => values.mean(),
            Self::Median => values.median(),
            Self::Min => values.min(),
            Self::Max => values.max(),
            Self::Std => values.std(),
            Self::Delta => values.delta(),
            Self::Count => Some(values.len() as f64),
        }
    }
}

#[derive(Deserialize)]
struct AggSlicesKwargs {
    aggregation: Aggregation,
    slices_include: Option<Vec<Range>>,
    slices_exclude: Option<Vec<Range>>,
}

/// Aggregate the values whose paired index falls within the requested
/// index ranges.
///
/// Missing data follows the polars convention: null elements are
/// skipped (pairwise with their index), while NaN is a legitimate
/// float value that flows into the aggregation. An empty selection
/// yields null (`count`: 0). NaN indices never match any range.
#[polars_expr(output_type=Float64)]
fn agg_slices(
    inputs: &[Series],
    kwargs: AggSlicesKwargs,
) -> PolarsResult<Series> {
    let li = ListInputs::with_inner_nulls(inputs)?;

    // No include ranges means include everything; excludes then carve it up.
    let mut slice = match kwargs.slices_include {
        None => Slice::full(),
        Some(ranges) => ranges.into_iter().fold(Slice::empty(), |s, r| s.include(r)),
    };
    for range in kwargs.slices_exclude.unwrap_or_default() {
        slice = slice.exclude(range);
    }

    let out: Float64Chunked = (0..li.len())
        .map(|i| {
            let Some(row) = li.row(i) else { return Ok(None) };
            let (values, indices) = (row[0], row[1]);
            polars_ensure!(
                values.len() == indices.len(),
                ComputeError: "agg_slices: value and index lists differ in length ({} vs {})",
                values.len(), indices.len()
            );

            // Polars missing-data convention: null elements are skipped
            // (pairwise with their index), NaN values flow through. The
            // aggregations turn an empty selection into None (count: 0).
            let selected = values
                .iter()
                .zip(indices)
                .filter_map(|(&value, &index)| Some((value?, index?)))
                .filter(|&(_, index)| slice.contains(index))
                .map(|(value, _index)| value)
                .collect::<Vec<f64>>();

            Ok(kwargs.aggregation.apply(&selected))
        })
        .collect::<PolarsResult<_>>()?;

    Ok(out.into_series())
}
