use butterworth::{Cutoff, Filter};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BandpassError {
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
pub fn bandpass(
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
