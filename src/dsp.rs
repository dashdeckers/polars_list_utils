use crate::numpy::interp;
use crate::util::same_dtype;
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use crate::dsp_util::{
    fft,
    fft_freqs,
    fft_normalized_freqs,
    bandpass,
    BandpassError,
    hanning_window,
    hamming_window,
};

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
