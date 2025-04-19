mod bandpass;
mod fft;
mod window;

pub use bandpass::{BandpassError, bandpass};
pub use fft::{fft, fft_freqs, fft_freqs_linspace};
pub use window::{hamming_window, hann_window};

use num_traits::Float;

pub fn linspace<T: Float + num_traits::NumCast>(
    start: T,
    stop: T,
    num: usize,
    endpoint: bool,
) -> Vec<T> {
    let to_float = |i: usize| T::from(i).unwrap_or_else(|| panic!());
    if endpoint {
        let dx = (stop - start) / to_float(num);
        (0..=num).map(|i| start + to_float(i) * dx).collect()
    } else {
        let dx = (stop - start) / to_float(num - 1);
        (0..num).map(|i| start + to_float(i) * dx).collect()
    }
}
