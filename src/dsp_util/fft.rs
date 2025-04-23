use crate::dsp_util::linspace;
use pyo3::prelude::*;
use realfft::RealFftPlanner;

/// Calculates the "real" FFT for the given "real-valued" input signal.
///
/// The first index corresponds to the DC component and the last index to
/// the Nyquist frequency.
///
/// ## Parameters
/// - `signal`: Array containing the discretized signal. Each value must be a
///   regular floating point number (no NaN or infinite) and the length must be
///   a power of two. Otherwise, the function panics.
///
/// ## Return value
/// New [Vec<f64>] of length `signal.len() / 2 + 1` with the result of the FFT.
///
/// ## Panics
/// The function panics if the length of the signal is not a power of two, or if
/// the signal contains NaN or infinite values.
///
/// ## More info
/// * <https://docs.rs/realfft/3.4.0/realfft/index.html>
pub fn fft(signal: &[f64]) -> Vec<f64> {
    // Ensure the signal length is a power of two
    let samples_len = signal.len();
    assert!(samples_len.is_power_of_two());

    // Create the FFT planner
    let mut real_planner = RealFftPlanner::<f64>::new();
    let r2c = real_planner.plan_fft_forward(samples_len);

    // Compute the FFT
    let mut spectrum = r2c.make_output_vec();
    r2c.process(&mut signal.to_owned(), &mut spectrum).unwrap();

    // Take only the real part of the complex FFT output and maybe normalize amplitude
    spectrum.iter().map(|val| val.norm()).collect()
}

/// Calculate the frequency values corresponding to the result of [fft].
///
/// This works for FFTs with "real-valued-inputs", without a complex part.
/// Basically, we throw away the second half of the FFT output for you, because
/// for real-valued inputs it is simply a mirror of the first half.
///
/// ## Parameters
/// - `n`: The number of frequency bins, i.e. the length of the FFT output.
/// - `fs`: sampling_rate, e.g. `44100 [Hz]`
///
/// ## Return value
/// New [Vec<f64>] with the frequency values in Hertz.
///
/// ## More info
/// * <https://stackoverflow.com/questions/4364823/>
#[rustfmt::skip]
#[allow(non_snake_case)]
#[pyfunction]
pub fn fft_freqs(
    n: usize,
    fs: usize,
) -> Vec<f64> {
    let fs = fs as f64;
    let N = (n - 1) * 2; // FFT output length including complex part
    (0..n)
        .map(|i| {
            (i as f64) * fs / (N as f64)
        })
        .collect()
}

/// Calculate an array of evenly spaced values to interpolate to.
///
/// This is a thin wrapper around a `numpy.linspace` equivalent function, and
/// generates a vector of `fnum` evenly spaced values from `0` to `fmax`. It
/// includes the `fmax` value.
///
/// ## Parameters
/// - `fmax`: Last value of the array.
/// - `fnum`: Number of values to generate.
///
/// ## Return value
/// New [Vec<f64>] with the generated values.
#[rustfmt::skip]
#[pyfunction]
pub fn fft_freqs_linspace(
    fnum: usize,
    fmax: f64,
) -> Vec<f64> {
    linspace(
        0 as f64,
        fmax,
        fnum,
        false,
    )
}
