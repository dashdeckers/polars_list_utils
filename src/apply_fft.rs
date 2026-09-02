use crate::transform::{Scaling, Window, compute_fft};
use crate::util::{Container, apply_list_transform, require_float_inner};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct FftKwargs {
    sample_rate: f64,
    window: Option<Window>,
    scaling: Option<Scaling>,
}

/// The FFT needs a power-of-two signal length of at least 2 (a
/// length-1 signal *is* a power of two, but the windows sum to zero on
/// it and the scaled spectrum reads NaN; a one-sample spectrum is
/// meaningless anyway). For an `Array` the width sits in the schema, so
/// a violation raises here at plan time; a `List` is checked per row.
fn fft_width_ok(width: usize) -> bool {
    width >= 2 && width.is_power_of_two()
}

/// The only width-transforming function: a one-sided real FFT of `n`
/// samples emits `n/2 + 1` bins, so an `Array(w)` narrows to
/// `Array(w/2 + 1)`. The inner dtype follows the signal.
fn fft_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let (container, inner) =
        require_float_inner(input_fields[0].dtype(), "apply_fft: signal_column")?;
    let out = match container {
        Container::List => Container::List,
        Container::Array(w) => {
            polars_ensure!(
                fft_width_ok(w),
                ComputeError:
                "apply_fft: signal_column has Array width {w}; the FFT needs a \
                power-of-two length of at least 2"
            );
            Container::Array(w / 2 + 1)
        }
    };
    Ok(Field::new(PlSmallStr::from(""), out.dtype(inner)))
}

/// Apply a one-sided FFT to a list column of time-domain signals.
///
/// # Arguments (via kwargs)
/// - `sample_rate`: Sampling rate in Hz
/// - `window`: Optional window function ("hanning"/"hann", "blackman")
/// - `scaling`: Optional scaling ("amplitude", "amplitude_squared",
///   "spectrum", "density")
///
/// Emits N/2 + 1 values from DC to Nyquist. A signal length that is not
/// a power of two of at least 2 raises — wrong shape is structural, not
/// data. Non-finite samples are legitimate floats and propagate: every
/// bin sums all samples, so one NaN yields an all-NaN spectrum rather
/// than a null row that would launder invalid data into missing data.
#[polars_expr(output_type_func=fft_output)]
fn apply_fft(
    inputs: &[Series],
    kwargs: FftKwargs,
) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.sample_rate > 0.0,
        ComputeError: "apply_fft: sample_rate must be positive, got {}",
        kwargs.sample_rate
    );
    let window = kwargs.window.unwrap_or_default();
    let scaling = kwargs.scaling.unwrap_or_default();
    let (input, inner) =
        require_float_inner(inputs[0].dtype(), "apply_fft: signal_column")?;
    let out = match input {
        Container::List => Container::List,
        Container::Array(w) => Container::Array(w / 2 + 1),
    };

    apply_list_transform(inputs, "apply_fft", &[false], out, &inner, |cols| {
        polars_ensure!(
            fft_width_ok(cols[0].len()),
            ComputeError:
            "apply_fft: a row's signal has {} samples; the FFT needs a power-of-two \
            length of at least 2 (pre-filter with list.len() to drop such rows)",
            cols[0].len()
        );
        compute_fft(cols[0], kwargs.sample_rate, window, scaling).map(Some)
    })
}
