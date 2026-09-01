use crate::util::{apply_list_transform, list_f64_output};
use interp::{InterpMode, interp_slice};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct InterpKwargs {
    #[serde(default)]
    strict: bool,
}

/// Interpolate `y` values from `(x, y)` data onto new `xp` coordinates,
/// as in numpy.interp; `xp` values outside the data range clamp to the
/// first/last `y` value.
///
/// `x` must be non-decreasing for the result to be meaningful; with
/// `strict` the first violation raises (a NaN in `x` counts as one),
/// without it unsorted `x` silently interpolates garbage, as in numpy.
#[polars_expr(output_type_func=list_f64_output)]
fn apply_interp(
    inputs: &[Series],
    kwargs: InterpKwargs,
) -> PolarsResult<Series> {
    apply_list_transform(inputs, |row_idx, cols| {
        polars_ensure!(
            cols[0].len() == cols[1].len(),
            ComputeError: "apply_interp: x and y lists differ in length ({} vs {})",
            cols[0].len(), cols[1].len()
        );
        // Non-decreasing means each pair compares Less or Equal; a NaN
        // pair is incomparable and counts as a violation.
        let in_order = |w: &&[f64]| {
            matches!(
                w[0].partial_cmp(&w[1]),
                Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
            )
        };
        if kwargs.strict
            && let Some(w) = cols[0].windows(2).find(|w| !in_order(w))
        {
            polars_bail!(ComputeError:
                "apply_interp: x_column must be non-decreasing with strict=true, \
                found {} followed by {} at row {row_idx}",
                w[0], w[1]
            );
        }
        Ok(Some(interp_slice(
            cols[0],
            cols[1],
            cols[2],
            &InterpMode::FirstLast,
        )))
    })
}
