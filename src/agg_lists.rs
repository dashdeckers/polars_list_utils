//! `agg_lists` itself is pure polars (see the Python wrapper); this
//! module only holds the plugin backing its `strict` check, because a
//! pure-polars expression has no way to raise at run time.

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct CheckListLenKwargs {
    list_length: usize,
}

fn passthrough_output(input_fields: &[Field]) -> PolarsResult<Field> {
    Ok(input_fields[0].clone())
}

/// Identity pass-through that raises when any row's list is longer
/// than `list_length` (which `agg_lists` would otherwise silently
/// truncate to). Null rows are data and pass through.
#[polars_expr(output_type_func=passthrough_output)]
fn check_list_len(
    inputs: &[Series],
    kwargs: CheckListLenKwargs,
) -> PolarsResult<Series> {
    let s = &inputs[0];
    for (i, row) in s.list()?.into_iter().enumerate() {
        if let Some(row) = row {
            polars_ensure!(
                row.len() <= kwargs.list_length,
                ComputeError:
                "agg_lists: row {i} has list length {} but list_length={} (strict=true)",
                row.len(), kwargs.list_length
            );
        }
    }
    Ok(s.clone())
}
