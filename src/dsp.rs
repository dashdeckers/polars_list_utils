use crate::dsp_util::{BandpassError, bandpass, fft, hamming_window, hann_window};
use crate::util::list_f64_dtype;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct ApplyFftKwargs {
    sample_rate: usize,
    window: Option<String>,
    bp_min: Option<f64>,
    bp_max: Option<f64>,
    bp_ord: Option<usize>,
    norm: Option<String>,
    skip_fft: bool,
}

/// Apply the (real) Fast Fourier Transform (FFT) to a `List` column of samples.
/// Optionally, apply a window function and/or a bandpass filter to the samples first.
///
/// The function raises an Error if:
/// * there are invalid values in the samples (NaN or infinite)
/// * the length of the samples is not a power of two
/// * the bandpass arguments are incorrect (e.g. min frequency is negative, ...)
///
/// ## Parameters
/// - `samples`: The `List` column of samples to apply the FFT to.
/// - `window`: Optional window function to apply to the samples before the FFT.
///   Possible values are "hanning" and "hamming".
/// - `bp_min`: Optional minimum frequency to let through the bandpass filter.
/// - `bp_max`: Optional maximum frequency to let through the bandpass filter.
/// - `bp_ord`: Optional order of the bandpass filter.
/// - `norm`: Mode of amplitude normalization. For real-valued inputs, we can
///   normalize the amplitude by dividing by $\frac{N}{2}$, where $N$ is the
///   length of the signal (mode="length"). If windowing is applied, we should
///   normalize the amplitude by dividing by the sum of the window function
///   (mode="window"). We can also skip amplitude normalization (mode=None).
///   Normalization here is meant to normalize the amplitude of the spectrum so
///   that the amplitude of the function in the time domain matches the amplitude
///   of the spectrum in the frequency domain.
/// - `skip_fft`: If true, the FFT is skipped and the transformed samples are
///   returned.
///
/// ## Panics
/// The function panics if there are invalid values in the `List` column.
///
/// ## Return value
/// New `List[f64]` column with the result of the FFT.
#[polars_expr(output_type_func=list_f64_dtype)]
fn expr_fft(
    inputs: &[Series],
    kwargs: ApplyFftKwargs,
) -> PolarsResult<Series> {
    let input = inputs[0].cast(&DataType::List(Box::new(DataType::Float64)))?;
    let ca = input.list()?;

    let mut invalid_value_encountered = false;
    let mut not_a_power_of_two = false;
    let mut bandpass_error: Option<BandpassError> = None;
    let dummy_vec: Vec<f64> = Vec::new();

    // TODO: This is a bit ugly, but we want to return a nice Error somehow.
    // probably this does a lot of uneccecary work if we do raise an Error
    let out: ListChunked = ca.apply_amortized(|s| {
        let s: &Series = s.as_ref();
        let ca: &Float64Chunked = s.f64().unwrap();

        // Collect the `List` values into a Vec<f64>
        // We remember to raise an Error on invalid values
        let samples: Vec<f64> = ca
            .iter()
            .map(|val| {
                if val.is_none_or(|x| x.is_nan() || x.is_infinite()) {
                    invalid_value_encountered = true;
                }
                // Default doesn't matter, we Err if the value is None anyway
                val.unwrap_or_default()
            })
            .collect();

        // We don't need further calculations if we raise an Error
        if invalid_value_encountered {
            return Series::new(PlSmallStr::EMPTY, dummy_vec.clone());
        }

        // We don't need further calculations if we raise an Error
        if !samples.len().is_power_of_two() {
            not_a_power_of_two = true;
            return Series::new(PlSmallStr::EMPTY, dummy_vec.clone());
        }

        // Maybe apply a bandpass filter to the samples
        let samples = bandpass(
            &samples,
            kwargs.sample_rate,
            kwargs.bp_ord.unwrap_or(4),
            kwargs.bp_min,
            kwargs.bp_max,
        );

        // We don't need further calculations if we raise an Error
        let mut samples = if let Err(err) = samples {
            bandpass_error = Some(err);
            return Series::new(PlSmallStr::EMPTY, dummy_vec.clone());
        } else {
            samples.unwrap()
        };

        // Maybe apply a window function to the samples
        let window_func = match kwargs.window.as_deref() {
            Some("hanning") => hann_window,
            Some("hamming") => hamming_window,
            _ => |n: usize| vec![1.0; n],
        };
        if kwargs.window.is_some() {
            samples = window_func(samples.len())
                .iter()
                .zip(samples.iter())
                .map(|(w, s)| s * w)
                .collect::<Vec<f64>>();
        }

        // If we skip the FFT, we return the samples now
        if kwargs.skip_fft {
            return Series::new(PlSmallStr::EMPTY, samples);
        }

        // Calculate the FFT
        let fft = fft(&samples);

        // Calculate the normalization factor for the FFT
        let normalization_factor = match kwargs.norm.as_deref() {
            // Instead of N, we use N/2+1 because we use RealFFT which ignores
            // the negative frequencies but includes the DC component
            Some("length") => (samples.len() as f64) / 2.0 + 1.0,
            Some("window") => window_func(samples.len() / 2 + 1).iter().sum(),
            _ => 1.0,
        };

        Series::new(
            PlSmallStr::EMPTY,
            fft.iter()
                .map(|val| val / normalization_factor)
                .collect::<Vec<f64>>(),
        )
    });

    if invalid_value_encountered {
        Err(PolarsError::ComputeError(
            "(apply_fft): Invalid value in the samples".into(),
        ))
    } else if not_a_power_of_two {
        Err(PolarsError::ComputeError(
            "(apply_fft): Length of the samples is not a power of two".into(),
        ))
    } else if bandpass_error.is_some() {
        Err(PolarsError::ComputeError(
            format!("(apply_fft): {}", bandpass_error.unwrap()).into(),
        ))
    } else {
        Ok(out.into_series())
    }
}
