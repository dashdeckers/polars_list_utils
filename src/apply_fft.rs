use crate::transform::{Scaling, Window, compute_fft};
use crate::util::{Container, apply_list_transform, transform_output};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

/// The only width-transforming function: a one-sided real FFT of `n`
/// samples emits `n/2 + 1` bins, so an `Array(w)` narrows to
/// `Array(w/2 + 1)`.
fn fft_output(input_fields: &[Field]) -> PolarsResult<Field> {
    transform_output(input_fields, 0, "apply_fft: signal_column", |w| w / 2 + 1)
}

#[derive(Deserialize)]
struct FftKwargs {
    sample_rate: f64,
    window: Option<Window>,
    scaling: Option<Scaling>,
}

/// Apply a one-sided FFT to a list column of time-domain signals.
///
/// # Arguments (via kwargs)
/// - `sample_rate`: Sampling rate in Hz
/// - `window`: Optional window function ("hann"/"hanning", "blackman")
/// - `scaling`: Optional scaling ("amplitude", "power", "psd")
///
/// Emits N/2 + 1 values from DC to Nyquist. Rows whose signal is not a
/// power-of-two length, or contains non-finite values, yield null.
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
    let (input, _) = Container::split(inputs[0].dtype(), "apply_fft: signal_column")?;
    let out = match input {
        Container::List => Container::List,
        Container::Array(w) => Container::Array(w / 2 + 1),
    };

    apply_list_transform(inputs, out, |cols| {
        Ok(compute_fft(cols[0], kwargs.sample_rate, window, scaling))
    })
}
