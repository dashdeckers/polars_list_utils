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
            Filter::Bandpass(..) => 2 * order,
            _ => order,
        }
    }

    /// Minimum sample count accepted by `bidirectional`, which reflects
    /// `3 * effective_order` samples of padding onto each end of the
    /// signal (and panics on exactly that length, hence the `+ 1`).
    pub(crate) fn min_samples(
        self,
        order: usize,
    ) -> usize {
        3 * self.effective_order(order) + 1
    }

    /// Validate parameters and design the Butterworth filter once.
    ///
    /// Cutoffs outside `(0, sample_rate / 2)` are rejected by the
    /// butterworth crate; order and cutoff ordering are checked here
    /// because the crate panics instead of erroring on them.
    pub(crate) fn build(
        self,
        sample_rate: f64,
        order: usize,
    ) -> PolarsResult<ButterworthFilter> {
        polars_ensure!(
            order >= 1,
            ComputeError: "filter_order must be at least 1, got {order}"
        );
        if let Filter::Bandpass(low, high) = self {
            polars_ensure!(
                low < high,
                ComputeError: "bandpass requires min_freq < max_freq, got {low} >= {high}"
            );
        }
        ButterworthFilter::new(order, sample_rate, self.to_cutoff())
            .map_err(|e| polars_err!(ComputeError: "failed to create filter: {e:?}"))
    }
}
