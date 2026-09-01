use serde::Deserialize;

/// The aggregation vocabulary shared by `agg_slices` and `cum_agg_runs`
/// (and mirrored in pure polars by `agg_lists`). Every member is valid
/// for every aggregation-taking function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Aggregation {
    Sum,
    Mean,
    Median,
    Min,
    Max,
    Std,
    Delta,
    Count,
}

impl Aggregation {
    pub(crate) fn apply(
        &self,
        values: &[f64],
    ) -> Option<f64> {
        match self {
            Self::Sum => values.sum(),
            Self::Mean => values.mean(),
            Self::Median => values.median(),
            Self::Min => values.min(),
            Self::Max => values.max(),
            Self::Std => values.std(),
            Self::Delta => values.delta(),
            Self::Count => Some(values.len() as f64),
        }
    }
}

/// Statistics over slices of data.
pub(crate) trait Statistic {
    fn values(&self) -> &[f64];

    fn is_empty(&self) -> bool {
        self.values().is_empty()
    }

    /// Sum with polars' identity element: an empty selection sums to
    /// `+0.0` (polars itself is split between `+0.0` and `-0.0` here,
    /// and Rust's `Iterator::sum` starts from IEEE's additive identity
    /// `-0.0`; the library standardizes on `+0.0`).
    fn sum(&self) -> Option<f64> {
        if self.is_empty() {
            return Some(0.0);
        }
        Some(self.values().iter().sum())
    }

    fn median(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mut values = self.values().to_vec();
        values.sort_by(|a, b| crate::util::tot_cmp(*a, *b));
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            Some((values[mid - 1] + values[mid]) / 2.0)
        } else {
            Some(values[mid])
        }
    }

    fn mean(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.values().iter().sum::<f64>() / self.values().len() as f64)
    }

    /// Sample standard deviation (ddof = 1), matching polars' default.
    /// Fewer than two values yield `None`, as in polars.
    fn std(&self) -> Option<f64> {
        let n = self.values().len();
        if n < 2 {
            return None;
        }
        let mean = self.mean()?;
        let variance: f64 = self
            .values()
            .iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>()
            / (n - 1) as f64;
        Some(variance.sqrt())
    }

    // The NaN seed makes IEEE min/max skip NaN values unless every
    // value is NaN, matching polars.

    fn min(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.values().iter().cloned().fold(f64::NAN, f64::min))
    }

    fn max(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.values().iter().cloned().fold(f64::NAN, f64::max))
    }

    fn delta(&self) -> Option<f64> {
        Some(self.max()? - self.min()?)
    }
}

impl Statistic for [f64] {
    fn values(&self) -> &[f64] {
        self
    }
}
