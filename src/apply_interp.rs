use crate::util::{apply_list_transform, list_f64_output};
use interp::{InterpMode, interp_slice};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

/// Interpolate `y` values from `(x, y)` data onto new `xp` coordinates,
/// as in numpy.interp; `xp` values outside the data range clamp to the
/// first/last `y` value.
#[polars_expr(output_type_func=list_f64_output)]
fn apply_interp(inputs: &[Series]) -> PolarsResult<Series> {
    apply_list_transform(inputs, |cols| {
        polars_ensure!(
            cols[0].len() == cols[1].len(),
            ComputeError: "apply_interp: x and y lists differ in length ({} vs {})",
            cols[0].len(), cols[1].len()
        );
        Ok(Some(interp_slice(
            cols[0],
            cols[1],
            cols[2],
            &InterpMode::FirstLast,
        )))
    })
}
