use crate::numpy::{interp, linspace};
use crate::util::same_dtype;
use butterworth::{Cutoff, Filter};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use realfft::RealFftPlanner;
use serde::Deserialize;
use thiserror::Error;

#[derive(Deserialize)]
struct ApplyFftKwargs {
    sample_rate: usize,
    window: Option<String>,
    bp_min: Option<f64>,
    bp_max: Option<f64>,
    bp_ord: Option<usize>,
    skip_fft: bool,
}

/// Apply the (real) Fast Fourier Transform (FFT) to a `List[f64]` column of samples.
/// Optionally, apply a window function and/or a bandpass filter to the samples first.
///
/// The function raises an Error if:
/// * the samples column is not of type `List(Float64)`
/// * there are invalid values in the samples (NaN or infinite)
/// * the length of the samples is not a power of two
/// * the bandpass arguments are incorrect (e.g. min frequency is negative, ...)
///
/// ## Parameters
/// - `samples`: The `List[f64]` column of samples to apply the FFT to.
/// - `window`: Optional window function to apply to the samples before the FFT.
///   Possible values are "hanning" and "hamming".
/// - `bp_min`: Optional minimum frequency to let through the bandpass filter.
/// - `bp_max`: Optional maximum frequency to let through the bandpass filter.
/// - `bp_ord`: Optional order of the bandpass filter.
/// - `skip_fft`: If true, the FFT is skipped and the samples are returned as is.
///
/// ## Panics
/// The function panics if there are invalid values in the `List[f64]` column.
///
/// ## Return value
/// New `List[f64]` column with the result of the FFT.
#[polars_expr(output_type_func=same_dtype)]
fn expr_fft(
    inputs: &[Series],
    kwargs: ApplyFftKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    if ca.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", ca.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    let mut invalid_value_encountered = false;
    let mut not_a_power_of_two = false;
    let mut bandpass_error: Option<BandpassError> = None;
    let dummy_vec: Vec<f64> = Vec::new();

    // TODO: This is a bit ugly, but we want to return a nice Error somehow.
    // probably this does a lot of uneccecary work if we do raise an Error
    let out: ListChunked = ca.apply_amortized(|s| {
        let s: &Series = s.as_ref();
        let ca: &Float64Chunked = s.f64().unwrap();

        // Collect the `List[f64]` values into a Vec<f64>
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
        let samples = if let Err(err) = samples {
            bandpass_error = Some(err);
            return Series::new(PlSmallStr::EMPTY, dummy_vec.clone());
        } else {
            samples.unwrap()
        };

        // Maybe apply a window function to the samples
        let samples = match kwargs.window.as_deref() {
            Some("hanning") => hanning_window(&samples),
            Some("hamming") => hamming_window(&samples),
            _ => samples,
        };

        // Maybe calculate the FFT and return as Series
        Series::new(
            PlSmallStr::EMPTY,
            if kwargs.skip_fft {
                samples
            } else {
                fft(&samples)
            },
        )
    });

    if invalid_value_encountered {
        Err(PolarsError::ComputeError(
            "Invalid value in the samples".into(),
        ))
    } else if not_a_power_of_two {
        Err(PolarsError::ComputeError(
            "Length of the samples is not a power of two".into(),
        ))
    } else if bandpass_error.is_some() {
        Err(PolarsError::ComputeError(
            format!("{}", bandpass_error.unwrap()).into(),
        ))
    } else {
        Ok(out.into_series())
    }
}

#[derive(Deserialize)]
struct GetFreqsKwargs {
    sample_rate: usize,
}

/// Get the frequencies corresponding to the result of the FFT.
///
/// The function raises an Error if the FFT column is not of type `List(Float64)`.
///
/// ## Parameters
/// - `list_col`: The `List[f64]` column of FFT amplitudes to get the frequencies for.
/// - `sample_rate`: The sampling rate of the samples.
///
/// ## Return value
/// New `List[f64]` column with the frequencies corresponding to the result of the FFT.
#[polars_expr(output_type_func=same_dtype)]
fn expr_fft_freqs(
    inputs: &[Series],
    kwargs: GetFreqsKwargs,
) -> PolarsResult<Series> {
    let ca = inputs[0].list()?;

    if ca.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", ca.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    let out: ListChunked = ca.apply_amortized(|s| {
        let s: &Series = s.as_ref();
        let ca: &Float64Chunked = s.f64().unwrap();

        // Collect the `List[f64]` values into a Vec<f64>
        let fft: Vec<f64> = ca.iter().map(|val| val.unwrap_or_default()).collect();

        // Calculate the FFT frequencies and return as Series
        Series::new(PlSmallStr::EMPTY, fft_freqs(fft.len(), kwargs.sample_rate))
    });

    Ok(out.into_series())
}

/// Calculates the "real" FFT for the given input samples.
///
/// The first index corresponds to the DC component and the last index to
/// the Nyquist frequency.
///
/// ## Parameters
/// - `samples`: Array with samples. Each value must be a regular floating
///   point number (no NaN or infinite) and the length must be
///   a power of two. Otherwise, the function panics.
///
/// ## Return value
/// New [Vec<f64>] of length `samples.len() / 2 + 1` with the result of the FFT.
///
/// ## Panics
/// The function panics if the length of the samples is not a power of two.
///
/// ## More info
/// * <https://docs.rs/realfft/3.4.0/realfft/index.html>
fn fft(samples: &[f64]) -> Vec<f64> {
    // Ensure the samples length is a power of two
    let samples_len = samples.len();
    assert!(samples_len.is_power_of_two());

    // Create the FFT planner
    // TODO: This should be cached
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(samples_len);

    // Compute the FFT
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut samples.to_owned(), &mut spectrum).unwrap();

    // Take only the real part of the complex FFT output
    // TODO: val.norm() vs abs().re()?
    spectrum
        .iter()
        .map(|val| val.norm() / (samples_len as f64).sqrt())
        .collect()
}

/// Calculate the frequency values corresponding to the result of [fft].
///
/// This works for "real" FFTs, that ignore the complex conjugate.
///
/// ## Parameters
/// - `sample_len` Length of the FFT result, of which half is the relevant part.
/// - `sample_rate` sampling_rate, e.g. `44100 [Hz]`
///
/// ## Return value
/// New [Vec<f64>] with the frequency values in Hertz.
///
/// ## More info
/// * <https://stackoverflow.com/questions/4364823/>
#[rustfmt::skip]
fn fft_freqs(
    sample_len: usize,
    sample_rate: usize,
) -> Vec<f64> {
    let fs = sample_rate as f64;
    let n = sample_len as f64;
    (0..sample_len / 2 + 1)
        .map(|i| {
            (i as f64) * fs / n
        })
        .collect()
}

/// Normalize the result of the FFT to by some normalization column.
///
/// The function raises an Error if the FFT column is not of type `List(Float64)`
/// or the normalization column is not of type `Float64`.
///
/// It normalizes the FFT by the normalization column and interpolates the result
/// to the maximum value of the normalization column such that the same number of
/// values is returned as in the FFT column.
///
/// ## Parameters
/// - `list_column`: The `List[f64]` column of FFT amplitudes to normalize.
/// - `norm_column`: The `f64` column of values to normalize the FFT amplitudes by.
///
/// ## Return value
/// New `List[f64]` column with the frequencies corresponding to the result of the FFT.
#[polars_expr(output_type_func=same_dtype)]
fn expr_normalize_ffts(
    inputs: &[Series],
) -> PolarsResult<Series> {
    let fft = inputs[0].list()?;
    let norm_col = inputs[1].f64()?;
    let max_norm_val = norm_col.max().unwrap();

    if fft.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", fft.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }
    if norm_col.dtype() != &DataType::Float64 {
        let msg = format!("Expected `Float64`, got: {}", norm_col.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    let out: ListChunked = fft.zip_and_apply_amortized(norm_col, |ca_fft, norm| {
        if let (Some(ca_fft), Some(norm)) = (ca_fft, norm) {
            let fft: &Float64Chunked = ca_fft.as_ref().f64().unwrap();

            let xp: Vec<f64> = fft
                .iter()
                .map(|val| {
                    if let Some(val) = val {
                        val / norm
                    } else {
                        f64::NAN
                    }
                })
                .collect();

            let fp: Vec<f64> = fft.iter().map(|val| val.unwrap_or(f64::NAN)).collect();

            let x = fft_normalized_freqs(fp.len(), max_norm_val);

            // x= fft_normalized_freqs(fft_col.len(), max_norm_val)
            // xp= get_fft_freqs() / norm_col
            // fp= fft_col

            let interpolated = interp(&x, &xp, &fp, None, None, None);

            Some(Series::new(PlSmallStr::EMPTY, interpolated))
        } else {
            None
        }
    });

    Ok(out.into_series())
}

/// Get the normalized frequencies corresponding to the result of the FFT.
///
/// The function raises an Error if the FFT column is not of type `List(Float64)`
/// or the normalization column is not of type `Float64`.
///
/// It returns the normalized frequencies such that it has the same number of
/// values has the FFT column and interpolated to the maximum value of the
/// normalization column.
///
/// ## Parameters
/// - `list_column`: The `List[f64]` column of FFT amplitudes to normalize.
/// - `norm_column`: The `f64` column of values to normalize the FFT amplitudes by.
///
/// ## Return value
/// New `List[f64]` column with the normalized frequencies corresponding to the result of the FFT.
#[polars_expr(output_type_func=same_dtype)]
fn expr_fft_normalized_freqs(
    inputs: &[Series],
) -> PolarsResult<Series> {
    let fft = inputs[0].list()?;
    let norm_col = inputs[1].f64()?;
    let max_norm_val = norm_col.max().unwrap();

    if fft.dtype() != &DataType::List(Box::new(DataType::Float64)) {
        let msg = format!("Expected `List(Float64)`, got: {}", fft.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }
    if norm_col.dtype() != &DataType::Float64 {
        let msg = format!("Expected `Float64`, got: {}", norm_col.dtype());
        return Err(PolarsError::ComputeError(msg.into()));
    }

    let out: ListChunked = fft.apply_amortized(|s| {
        let s: &Series = s.as_ref();
        let ca: &Float64Chunked = s.f64().unwrap();

        // Collect the `List[f64]` values into a Vec<f64>
        let fft: Vec<f64> = ca.iter().map(|val| val.unwrap_or_default()).collect();

        // Calculate the FFT frequencies and return as Series
        Series::new(
            PlSmallStr::EMPTY,
            fft_normalized_freqs(fft.len(), max_norm_val),
        )
    });

    Ok(out.into_series())
}

/// Calculate the normalized frequency values corresponding to the result of [fft].
///
/// This works for "real" FFTs, that ignore the complex conjugate.
///
/// ## Parameters
/// - `fft_len` Length of the FFT result, of which everything (!) is relevant.
/// - `max_norm_val`: The maximum value to normalize the FFT amplitudes to.
///
/// ## Return value
/// New [Vec<f64>] with the normalized frequency values in Hertz.
///
/// ## More info
/// * <https://stackoverflow.com/questions/4364823/>
#[rustfmt::skip]
fn fft_normalized_freqs(
    fft_len: usize,
    max_norm_val: f64,
) -> Vec<f64> {
    let (samples, _) = linspace(
        0 as f64,
        max_norm_val,
        fft_len,
        true,
        false,
    );

    samples
}

/// Applies the Hann window to an array of samples.
///
/// ## Return value
/// New [Vec<f64>] with the result of the Hann window applied to the sample array.
///
/// ## More info
/// * <https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows>
#[rustfmt::skip]
fn hanning_window(
    samples: &[f64],
) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    let n = samples.len() as f64;
    samples
        .iter()
        .enumerate()
        .map(|(i, sample)| {
            0.5 * (1.0 - (2.0 * pi * (i as f64) / n).cos()) * sample
        })
        .collect()
}

/// Applies a Hamming window to an array of samples.
///
/// ## Return value
/// New [Vec<f64>] with the result of the Hamming window applied to the sample array.
///
/// ## More info
/// * <https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows>
#[rustfmt::skip]
fn hamming_window(
    samples: &[f64],
) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    let n = samples.len() as f64;
    samples
        .iter()
        .enumerate()
        .map(|(i, sample)| {
            0.54 - (0.46 * (2.0 * pi * (i as f64) / (n - 1.0)).cos()) * sample
        })
        .collect()
}

#[derive(Debug, Error)]
enum BandpassError {
    #[error("Minimum frequency is negative")]
    MinNegative,
    #[error("Maximum frequency is lower than the minimum frequency")]
    MaxLessThanMin,
    #[error("Maximum frequency is larger than the Nyquist frequency")]
    MaxLargerThanNyquist,
}

/// Applies a bandpass filter to an array of samples.
///
/// This function applies some extra logic to handle the edge cases where the
/// minimum frequency is zero or the maximum frequency is the Nyquist frequency:
/// - If the minimum frequency is zero, a lowpass filter is applied. Zero is the
///   minimum relevant frequency.
/// - If the maximum frequency is larger than or equal to the Nyquist frequency,
///   a highpass filter is applied. Nyquist is the maximum relevant frequency.
/// - If both conditions are true, the samples are returned as is. No need to
///   apply a filter at all.
/// - Otherwise a bandpass filter is applied.
///
/// ## Parameters
/// - `samples`: Array with samples. Each value must be a regular f64 (no NaN or infinite).
/// - `sample_rate` sampling_rate, e.g. `44100 [Hz]`
/// - `order`: The order of the filter.
/// - `min`: The minimum frequency let through by the bandpass filter.
/// - `max`: The maximum frequency let through by the bandpass filter.
///
/// ## Return value
/// New [Vec<f64>] with the result of the bandpass filter applied to the sample array.
///
/// ## Panics
/// The function panics if the min frequency is less than zero or the max frequency
/// is lower than the min frequency or the max frequency is larger than the Nyquist
/// frequency.
///
/// ## More info
/// * <https://docs.rs/butterworth/0.1.0/butterworth/index.html>
fn bandpass(
    samples: &[f64],
    sample_rate: usize,
    order: usize,
    min: Option<f64>,
    max: Option<f64>,
) -> Result<Vec<f64>, BandpassError> {
    let nyquist = sample_rate as f64 / 2.0;

    // min and max are semantically None at 0.0 and the Nyquist frequency
    let min = min.unwrap_or(0.0);
    let max = max.unwrap_or(nyquist);

    // Ensure the min frequency is not negative
    if min < 0.0 {
        return Err(BandpassError::MinNegative);
    }

    // Ensure the max frequency is not lower than the min frequency
    if max < min {
        return Err(BandpassError::MaxLessThanMin);
    }

    // Ensure the max frequency is not larger than the Nyquist frequency
    if max > nyquist {
        return Err(BandpassError::MaxLargerThanNyquist);
    }

    // Set the cutoff frequencies
    let cutoff = if min == 0. && max == nyquist {
        return Ok(samples.to_owned());
    } else if min == 0. {
        Cutoff::LowPass(max)
    } else if max == nyquist {
        Cutoff::HighPass(min)
    } else {
        Cutoff::BandPass(min, max)
    };

    // Assuming the sample rate is as given, design an nth order cutoff filter.
    let filter = Filter::new(order, sample_rate as f64, cutoff).unwrap();

    // Apply a bidirectional filter to the data
    Ok(filter.bidirectional(&samples.to_owned()).unwrap())

    // // Manually specify a padding length if the default behavior of SciPy is desired
    // filter.bidirectional_with_padding(
    //     &samples.to_owned(),
    //     3 * (filter.order() + 1),
    // ).unwrap()
}
