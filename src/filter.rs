use butterworth::{Cutoff, Filter as ButterworthFilter};
use polars::prelude::*;

/// Butterworth filter with cutoff frequency/frequencies.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Filter {
    /// High-pass: passes frequencies above cutoff.
    Highpass(f64),
    /// Low-pass: passes frequencies below cutoff.
    Lowpass(f64),
    /// Band-pass: passes frequencies between low and high cutoffs.
    Bandpass(f64, f64),
}

impl Filter {
    fn to_cutoff(self) -> Cutoff {
        match self {
            Filter::Highpass(freq) => Cutoff::HighPass(freq),
            Filter::Lowpass(freq) => Cutoff::LowPass(freq),
            Filter::Bandpass(low, high) => Cutoff::BandPass(low, high),
        }
    }

    /// The design order the butterworth crate actually applies
    /// (doubled for bandpass, as in scipy.signal.butter).
    fn effective_order(
        self,
        order: usize,
    ) -> usize {
        match self {
            Filter::Bandpass(..) => order.saturating_mul(2),
            _ => order,
        }
    }

    /// Minimum sample count accepted by `bidirectional`, which reflects
    /// `3 * effective_order` samples of padding onto each end of the
    /// signal (and panics on exactly that length, hence the `+ 1`).
    ///
    /// Saturating: release builds wrap on overflow, so an absurd
    /// `filter_order` could otherwise wrap this to a tiny number and
    /// arithmetically defeat the very guard that keeps the crate from
    /// indexing past the end of the data.
    pub(crate) fn min_samples(
        self,
        order: usize,
    ) -> usize {
        self.effective_order(order)
            .saturating_mul(3)
            .saturating_add(1)
    }

    /// Validate parameters and design the Butterworth filter once.
    ///
    /// All parameter errors carry the function name and the offending
    /// keyword. The cutoff range is checked here rather than left to
    /// the butterworth crate, whose errors name neither and would leak
    /// a transitive dependency's Debug enum into a user-facing message
    /// (the `map_err` below is an unreachable-in-practice backstop). A
    /// NaN cutoff fails the range check and raises too.
    pub(crate) fn build(
        self,
        sample_rate: f64,
        order: usize,
    ) -> PolarsResult<ButterworthFilter> {
        polars_ensure!(
            order >= 1,
            ComputeError: "apply_butterworth: filter_order must be at least 1, got {order}"
        );
        let nyquist = sample_rate / 2.0;
        let in_range = |name: &str, freq: f64| -> PolarsResult<()> {
            polars_ensure!(
                freq > 0.0 && freq < nyquist,
                ComputeError:
                "apply_butterworth: {name} must be in (0, sample_rate/2 = {nyquist}), \
                got {freq}"
            );
            Ok(())
        };
        match self {
            Filter::Highpass(freq) => in_range("min_freq", freq)?,
            Filter::Lowpass(freq) => in_range("max_freq", freq)?,
            Filter::Bandpass(low, high) => {
                in_range("min_freq", low)?;
                in_range("max_freq", high)?;
                polars_ensure!(
                    low < high,
                    ComputeError:
                    "apply_butterworth: bandpass requires min_freq < max_freq, \
                    got {low} >= {high}"
                );
            }
        }
        ButterworthFilter::new(order, sample_rate, self.to_cutoff()).map_err(
            |e| polars_err!(ComputeError: "apply_butterworth: failed to create filter: {e:?}"),
        )
    }
}
