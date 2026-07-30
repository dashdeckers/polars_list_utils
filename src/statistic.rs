/// Statistics over slices of data.
pub(crate) trait Statistic {
    fn values(&self) -> &[f64];

    fn is_empty(&self) -> bool {
        self.values().is_empty()
    }

    fn median(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mut values = self.values().to_vec();
        values.sort_by(|a, b| a.total_cmp(b));
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
