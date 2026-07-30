use crate::transform::{Scaling, Window, compute_fft};
use crate::util::{apply_list_transform, list_f64_output};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

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
#[polars_expr(output_type_func=list_f64_output)]
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

    apply_list_transform(inputs, |cols| {
        Ok(compute_fft(cols[0], kwargs.sample_rate, window, scaling))
    })
}
