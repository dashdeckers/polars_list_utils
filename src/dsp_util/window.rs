/// Applies the Hann window to an array of samples.
///
/// ## Return value
/// New [Vec<f64>] with the result of the Hann window applied to the sample array.
///
/// ## More info
/// * <https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows>
#[rustfmt::skip]
pub fn hanning_window(
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
pub fn hamming_window(
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
