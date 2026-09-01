use crate::util::{
    Container, apply_list_transform, ensure_equal_widths, transform_output,
};
use interp::{InterpMode, interp_slice};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct InterpKwargs {
    #[serde(default)]
    strict: bool,
}

/// Shape follows `xp`: the output holds one value per query coordinate,
/// so the query column decides both length and container. The `x`/`y`
/// containers do not reach the output — but they are still split here,
/// because that is the only thing standing between a zero-width `Array`
/// and the polars-arrow panic, and because an `x`/`y` width mismatch is
/// decidable from the schema.
fn interp_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let (x, _) = Container::split(input_fields[0].dtype(), "apply_interp: x_column")?;
    let (y, _) = Container::split(input_fields[1].dtype(), "apply_interp: y_column")?;
    ensure_equal_widths(x, y, "apply_interp: x and y Arrays must have equal widths")?;
    transform_output(input_fields, 2, "apply_interp: xp_column", |w| w)
}

/// Interpolate `y` values from `(x, y)` data onto new `xp` coordinates,
/// as in numpy.interp; `xp` values outside the data range clamp to the
/// first/last `y` value.
///
/// `x` must be non-decreasing for the result to be meaningful; with
/// `strict` the first descending pair raises, without it unsorted `x`
/// silently interpolates garbage, as in numpy. NaN in `x` is a
/// legitimate float that propagates into the output either way: the
/// check exists for the silent failure, not the visible one.
#[polars_expr(output_type_func=interp_output)]
fn apply_interp(
    inputs: &[Series],
    kwargs: InterpKwargs,
) -> PolarsResult<Series> {
    let (out, _) = Container::split(inputs[2].dtype(), "apply_interp: xp_column")?;
    apply_list_transform(inputs, out, |cols| {
        polars_ensure!(
            cols[0].len() == cols[1].len(),
            ComputeError: "apply_interp: x and y lists differ in length ({} vs {})",
            cols[0].len(), cols[1].len()
        );
        // Compare each coordinate against the last non-NaN one rather
        // than its immediate neighbour. NaN stays a legitimate float
        // that propagates into the output, but it must not hide a
        // descent: every IEEE comparison with NaN is false, so an
        // adjacent-pair test reads [0, 10, NaN, 1] as sorted.
        //
        // No row index in the message: the plugin is handed one chunk
        // at a time, so any position named here would be chunk-local.
        if kwargs.strict {
            let mut last: Option<f64> = None;
            for &x in cols[0] {
                if x.is_nan() {
                    continue;
                }
                if let Some(prev) = last
                    && x < prev
                {
                    polars_bail!(ComputeError:
                        "apply_interp: x_column must be non-decreasing with \
                        strict=true, found {prev} followed by {x}"
                    );
                }
                last = Some(x);
            }
        }
        Ok(Some(interp_slice(
            cols[0],
            cols[1],
            cols[2],
            &InterpMode::FirstLast,
        )))
    })
}
