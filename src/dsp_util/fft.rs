use crate::numpy::linspace;
use realfft::RealFftPlanner;

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
pub fn fft(samples: &[f64]) -> Vec<f64> {
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
pub fn fft_freqs(
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
pub fn fft_normalized_freqs(
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
