use crate::filter::Filter;
use crate::util::{apply_list_transform, list_f64_output};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct ButterworthKwargs {
    sample_rate: f64,
    min_freq: Option<f64>,
    max_freq: Option<f64>,
    filter_order: Option<usize>,
}

/// Apply a zero-phase Butterworth filter to a list column.
///
/// # Arguments (via kwargs)
/// - `sample_rate`: Sampling rate in Hz
/// - `min_freq`: Highpass cutoff frequency (Hz)
/// - `max_freq`: Lowpass cutoff frequency (Hz)
/// - `filter_order`: Filter order (default 4)
///
/// If both cutoffs are specified, a bandpass filter is applied; with one,
/// a highpass or lowpass; with neither, the input passes through.
///
/// Filtering is bidirectional (filtfilt): zero phase lag, squared
/// magnitude response, cutoff at -6 dB. Bandpass doubles the design
/// order, as in scipy.signal.butter. Rows with no more than 3x the
/// effective order samples (the reflection padding) yield null.
#[polars_expr(output_type_func=list_f64_output)]
fn apply_butterworth(
    inputs: &[Series],
    kwargs: ButterworthKwargs,
) -> PolarsResult<Series> {
    let filter = match (kwargs.min_freq, kwargs.max_freq) {
        (Some(lo), Some(hi)) => Filter::Bandpass(lo, hi),
        (Some(lo), None) => Filter::Highpass(lo),
        (None, Some(hi)) => Filter::Lowpass(hi),
        (None, None) => {
            return apply_list_transform(inputs, |_row, cols| Ok(Some(cols[0].to_vec())));
        }
    };
    let order = kwargs.filter_order.unwrap_or(4);
    let built = filter.build(kwargs.sample_rate, order)?;
    let min_samples = filter.min_samples(order);

    apply_list_transform(inputs, move |_row, cols| {
        if cols[0].len() < min_samples {
            return Ok(None);
        }
        Ok(built.bidirectional(&cols[0].to_vec()).ok())
    })
}
