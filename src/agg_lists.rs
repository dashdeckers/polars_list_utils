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

/// Identity pass-through that raises when a row would lose data to
/// `agg_lists`' silent truncation at `list_length`.
///
/// Being merely longer than the window is not enough: every
/// aggregation skips null elements, so truncating a null-padded tail
/// is provably lossless, and raising on it would be a false positive
/// whose only remedy is turning the check off entirely. Fixed-capacity
/// buffers and ragged data widened to a common length are ordinary
/// inputs here. Null rows are data and pass through.
///
/// The message names no row index: the plugin is handed one chunk at a
/// time, so any position named here would be chunk-local and point at
/// a different, innocent row of the frame.
#[polars_expr(output_type_func=passthrough_output)]
fn check_list_len(
    inputs: &[Series],
    kwargs: CheckListLenKwargs,
) -> PolarsResult<Series> {
    let s = &inputs[0];
    let width = kwargs.list_length;
    for row in s.list()?.into_iter().flatten() {
        if row.len() > width {
            let tail = row.slice(width as i64, row.len() - width);
            polars_ensure!(
                tail.null_count() == tail.len(),
                ComputeError:
                "agg_lists: a row has list length {} with non-null values past \
                list_length={} (strict=true)",
                row.len(), width
            );
        }
    }
    Ok(s.clone())
}
