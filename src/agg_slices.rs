use crate::slice::{Range, Slice};
use crate::statistic::{Aggregation, Empty};
use crate::util::{Container, ListInputs, ensure_equal_widths};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct AggSlicesKwargs {
    aggregation: Aggregation,
    slices_include: Option<Vec<Range>>,
    slices_exclude: Option<Vec<Range>>,
    empty: Empty,
}

/// Always a `Float64` scalar — there is no container to propagate — but
/// resolved through a function rather than declared statically, so the
/// inputs are validated at plan time. That is what keeps a zero-width
/// `Array` from reaching the FFI boundary, where polars would panic
/// rather than raise.
fn agg_slices_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let (values, _) =
        Container::split(input_fields[0].dtype(), "agg_slices: value_column")?;
    let (indices, _) =
        Container::split(input_fields[1].dtype(), "agg_slices: index_column")?;
    ensure_equal_widths(
        values,
        indices,
        "agg_slices: value and index Arrays must have equal widths",
    )?;
    Ok(Field::new(PlSmallStr::from(""), DataType::Float64))
}

/// Aggregate the values whose paired index falls within the requested
/// index ranges.
///
/// Missing data follows the polars convention: null elements are
/// skipped (pairwise with their index), while NaN is a legitimate
/// float value that flows into the aggregation. An empty selection
/// yields null (`count` reading 0, and `sum` following the `empty`
/// setting). NaN indices never match any range.
#[polars_expr(output_type_func=agg_slices_output)]
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
            // aggregations turn an empty selection into None, except
            // count, which reads 0.
            let selected = values
                .iter()
                .zip(indices)
                .filter_map(|(&value, &index)| Some((value?, index?)))
                .filter(|&(_, index)| slice.contains(index))
                .map(|(value, _index)| value)
                .collect::<Vec<f64>>();

            Ok(kwargs.aggregation.apply(&selected, kwargs.empty))
        })
        .collect::<PolarsResult<_>>()?;

    Ok(out.into_series())
}
