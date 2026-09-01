//! `agg_lists` itself is an expression composition over polars' own
//! vertical aggregations (see the Python wrapper) — it has to be, since
//! it reduces many rows to one and an elementwise plugin cannot. This
//! module holds the one plugin it needs: a pass-through that normalizes
//! the container and, under `strict`, validates the row width.
//!
//! Normalization is not optional. The `.list` namespace rejects `Array`
//! columns outright, so without this the composition would only accept
//! `List`; and the wrapper cannot branch on dtype itself, because an
//! expression is built before any schema is resolved.

use crate::util::{Container, as_list};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct PrepareKwargs {
    list_length: usize,
    strict: bool,
}

fn prepare_output(
    input_fields: &[Field],
    kwargs: PrepareKwargs,
) -> PolarsResult<Field> {
    let (container, inner) =
        Container::split(input_fields[0].dtype(), "agg_lists: list_column")?;
    // An Array's width is in the schema, so a mismatch against
    // list_length is decidable before execution (validation, not
    // inference: the signature does not fork on container).
    if let Container::Array(width) = container {
        polars_ensure!(
            width == kwargs.list_length,
            ComputeError:
            "agg_lists: list_length={} does not match the Array width {width}",
            kwargs.list_length
        );
    }
    Ok(Field::new(
        PlSmallStr::from(""),
        DataType::List(Box::new(inner)),
    ))
}

/// Normalize `list_column` to a `List` column, and under `strict` raise
/// when a row would lose data to `agg_lists`' truncation at
/// `list_length`.
///
/// Being merely longer than the window is not enough: every aggregation
/// skips null elements, so truncating a null-padded tail is provably
/// lossless, and raising on it would be a false positive whose only
/// remedy is turning the check off entirely. Fixed-capacity buffers and
/// ragged data widened to a common length are ordinary inputs here.
/// Null rows are data and pass through.
///
/// The message names no row index: the plugin is handed one chunk at a
/// time, so any position named here would be chunk-local and point at a
/// different, innocent row of the frame.
#[polars_expr(output_type_func_with_kwargs=prepare_output)]
fn prepare_agg_lists(
    inputs: &[Series],
    kwargs: PrepareKwargs,
) -> PolarsResult<Series> {
    let s = as_list(&inputs[0])?;
    if !kwargs.strict {
        return Ok(s);
    }
    let width = kwargs.list_length;
    for row in s.list()?.into_iter().flatten() {
        if row.len() > width {
            let tail = row.slice(width as i64, row.len() - width);
            polars_ensure!(
                tail.null_count() == tail.len(),
                ComputeError:
                "agg_lists: a row has list length {} with non-null values past \
                list_length={width} (strict=true)",
                row.len()
            );
        }
    }
    Ok(s)
}
