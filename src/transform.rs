//! FFT of time-domain signals with standard windowing and scaling.
//!
//! AI DISCLAIMER: The documentation in this file is AI-generated.
//!
//! Conventions follow Heinzel, Rüdiger & Schilling (2002), "Spectrum and
//! spectral density estimation by the Discrete Fourier transform (DFT)",
//! and match scipy: `Amplitude` reads a tone's peak amplitude A, `Power`
//! its mean-square A²/2 (scipy.signal.periodogram scaling='spectrum'),
//! and `Psd` its power density (scaling='density').

use std::cell::RefCell;

use polars::prelude::*;
use realfft::RealFftPlanner;
use serde::Deserialize;

thread_local! {
    static FFT_PLANNER: RefCell<RealFftPlanner<f64>> = RefCell::new(RealFftPlanner::new());
}

/// Window function applied before the FFT, in periodic (DFT-even) form,
/// matching `scipy.signal.windows.hann(..., sym=False)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Window {
    #[default]
    None,
    /// Hann window: moderate frequency resolution and sidelobe suppression.
    #[serde(alias = "hann")]
    Hanning,
    /// Blackman window: better sidelobe suppression, wider main lobe.
    Blackman,
}

impl Window {
    /// Periodic window coefficients (denominator N, not N-1).
    fn coefficients(
        self,
        len: usize,
    ) -> Vec<f64> {
        use std::f64::consts::PI;
        let n = len as f64;
        (0..len)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / n;
                match self {
                    Window::None => 1.0,
                    // Hann: 0.5 - 0.5*cos(2πi/N)
                    Window::Hanning => 0.5 * (1.0 - x.cos()),
                    // Blackman: 0.42 - 0.5*cos(2πi/N) + 0.08*cos(4πi/N)
                    Window::Blackman => 0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos(),
                }
            })
            .collect()
    }
}

/// Scaling of the one-sided spectrum, in scipy's vocabulary.
///
/// The names follow `scipy.signal.periodogram` because the previous set
/// produced a real misreading: `"power"` was read as A² when it yields
/// the mean-square A²/2. `"spectrum"` and `"density"` are scipy's
/// unambiguous names for those quantities; `"amplitude_squared"` is the
/// convenience the misreading was reaching for, defined as the square
/// of the `"amplitude"` scaling. The old names are removed, not
/// aliased — the ambiguous name disappearing is the point.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Scaling {
    /// Raw FFT magnitudes of the windowed signal.
    #[default]
    None,
    /// Peak-amplitude spectrum: a tone of amplitude A reads A.
    Amplitude,
    /// `amplitude`² per bin (one-sided doubling squared along with it):
    /// a tone of amplitude A reads A². A convenience with no external
    /// anchor — elementwise squaring of a list column is exactly the
    /// boilerplate this library exists to remove.
    AmplitudeSquared,
    /// Power spectrum: a tone of amplitude A reads its mean-square
    /// A²/2, as `scipy.signal.periodogram(scaling="spectrum")`.
    Spectrum,
    /// Power spectral density in [unit²/Hz]: power divided by the
    /// window's equivalent noise bandwidth. Integrates to the signal's
    /// mean-square power (Parseval), as `scaling="density"`.
    Density,
}

/// Raw one-sided FFT magnitudes (N/2 + 1 values from DC to Nyquist),
/// using a thread-local planner to cache FFT plans across rows.
///
/// A failed plan or process call raises: it is a bug surfacing, not
/// missing data, and converting it to a null row would launder an
/// internal error into "missing signal".
fn fft_magnitudes(samples: &[f64]) -> PolarsResult<Vec<f64>> {
    FFT_PLANNER.with(|cell| {
        let fft = cell.borrow_mut().plan_fft_forward(samples.len());
        let mut input = samples.to_vec();
        let mut spectrum = fft.make_output_vec();
        fft.process(&mut input, &mut spectrum).map_err(
            |e| polars_err!(ComputeError: "apply_fft: FFT kernel failed: {e}"),
        )?;
        Ok(spectrum.iter().map(|c| c.norm()).collect())
    })
}

/// One-sided spectrum of `signal` with the given window and scaling.
///
/// The caller guarantees a power-of-two length of at least 2. Non-finite
/// samples are legitimate floats and flow through the arithmetic: every
/// bin sums all samples, so one NaN yields an all-NaN spectrum — invalid
/// data stays visibly invalid rather than becoming missing.
///
/// Scaling normalizes by S1 = Σw[n]; interior bins are doubled **in
/// each scaling's own domain** (×2 amplitude, ×2 power) to account for
/// the discarded negative frequencies, while DC and Nyquist (which
/// have no mirror bin) are not.
pub(crate) fn compute_fft(
    signal: &[f64],
    sample_rate: f64,
    window: Window,
    scaling: Scaling,
) -> PolarsResult<Vec<f64>> {
    let coeffs = window.coefficients(signal.len());
    let windowed: Vec<f64> = signal.iter().zip(&coeffs).map(|(v, c)| v * c).collect();
    let magnitudes = fft_magnitudes(&windowed)?;

    let s1: f64 = coeffs.iter().sum();
    let s2: f64 = coeffs.iter().map(|c| c * c).sum();
    // Equivalent noise bandwidth in Hz: ENBW_bins * (fs / N) = fs * S2 / S1².
    let enbw_hz = sample_rate * s2 / (s1 * s1);

    let last = magnitudes.len() - 1;
    Ok(magnitudes
        .iter()
        .enumerate()
        .map(|(i, &mag)| {
            // One-sided doubling: DC (i=0) and Nyquist (i=last) have
            // no negative-frequency counterpart and are not doubled.
            let sf = if i == 0 || i == last { 1.0 } else { 2.0 };
            match scaling {
                Scaling::None => mag,
                Scaling::Amplitude => sf * mag / s1,
                Scaling::AmplitudeSquared => (sf * mag / s1).powi(2),
                Scaling::Spectrum => sf * (mag / s1).powi(2),
                Scaling::Density => sf * (mag / s1).powi(2) / enbw_hz,
            }
        })
        .collect())
}
