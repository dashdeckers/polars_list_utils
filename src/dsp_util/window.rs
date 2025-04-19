/// The Hann window function.
///
/// ## Return value
/// New [Vec<f64>] with the Hann window.
///
/// ## More info
/// * <https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows>
#[rustfmt::skip]
pub fn hann_window(
    n: usize,
) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|i| {
            0.5 * (1.0 - (2.0 * pi * (i as f64) / (n as f64)).cos())
        })
        .collect()
}

/// The Hamming window function.
///
/// ## Return value
/// New [Vec<f64>] with the Hamming window.
///
/// ## More info
/// * <https://en.wikipedia.org/wiki/Window_function#Hann_and_Hamming_windows>
#[rustfmt::skip]
pub fn hamming_window(
    n: usize,
) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|i| {
            0.54 - (0.46 * (2.0 * pi * (i as f64) / ((n as f64) - 1.0)).cos())
        })
        .collect()
}
