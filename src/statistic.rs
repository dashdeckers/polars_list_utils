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

/// What an empty selection yields for `sum`, the only aggregation with
/// a non-null identity element.
///
/// Every other aggregation yields null over nothing under either
/// setting, and `count` yields 0 under either, so this switch reaches
/// exactly one cell of the matrix. It applies to a selection that came
/// out empty; a null row or an empty list never reaches this code,
/// having already short-circuited to a null output row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Empty {
    /// Summing nothing is unknown, not zero: the safer reading when an
    /// empty selection usually means a misconfigured range.
    #[default]
    Null,
    /// Polars' convention: the identity element of addition.
    Zero,
}

impl Aggregation {
    pub(crate) fn apply(
        &self,
        values: &[f64],
        empty: Empty,
    ) -> Option<f64> {
        match self {
            Self::Sum if values.is_empty() && empty == Empty::Zero => Some(0.0),
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

/// Running minimum that skips NaN and keeps the **first** of values
/// that compare equal, as polars does. `f64::min` would return the
/// later one, which flips the sign of a `-0.0`/`+0.0` tie.
pub(crate) fn min_fold(
    acc: f64,
    x: f64,
) -> f64 {
    if acc.is_nan() {
        x
    } else if x.is_nan() || acc <= x {
        acc
    } else {
        x
    }
}

/// Running maximum, first-wins on ties. See [`min_fold`].
pub(crate) fn max_fold(
    acc: f64,
    x: f64,
) -> f64 {
    if acc.is_nan() {
        x
    } else if x.is_nan() || acc >= x {
        acc
    } else {
        x
    }
}

/// The median of two middle order statistics, as polars computes it.
///
/// Linear interpolation rather than `(a + b) / 2`, which overflows for
/// large equal values, plus an equality short-circuit, without which
/// equal infinities would read NaN. Together these reproduce polars on
/// every probed input, including `-0.0`/`+0.0` ties (the short-circuit
/// returns the first, so a stable sort carries the sign through).
pub(crate) fn midpoint(
    a: f64,
    b: f64,
) -> f64 {
    if a == b { a } else { a + (b - a) * 0.5 }
}

/// Statistics over slices of data.
pub(crate) trait Statistic {
    fn values(&self) -> &[f64];

    fn is_empty(&self) -> bool {
        self.values().is_empty()
    }

    /// Sum of the selected values, or null when nothing was selected.
    ///
    /// A deliberate deviation from polars, which returns its identity
    /// element `0.0` there: summing nothing is not the same as summing
    /// to zero, and reporting a real zero for absent data is the more
    /// dangerous of the two errors. `count` keeps polars' `0`, which is
    /// the semantically correct answer to "how many values".
    ///
    /// A non-empty selection folds from `+0.0`, so an all-`-0.0` one
    /// reads `+0.0` as polars does; Rust's `Iterator::sum` would fold
    /// from IEEE's `-0.0` identity instead.
    fn sum(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.values().iter().fold(0.0, |acc, &v| acc + v))
    }

    fn median(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        let mut values = self.values().to_vec();
        // sort_by is stable, so ties keep their input order and the
        // midpoint short-circuit returns the earlier of a signed-zero
        // pair -- matching polars.
        values.sort_by(|a, b| crate::util::tot_cmp(*a, *b));
        let mid = values.len() / 2;
        if values.len().is_multiple_of(2) {
            Some(midpoint(values[mid - 1], values[mid]))
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

    // The NaN seed makes the folds skip NaN values unless every value
    // is NaN, matching polars.

    fn min(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.values().iter().cloned().fold(f64::NAN, min_fold))
    }

    fn max(&self) -> Option<f64> {
        if self.is_empty() {
            return None;
        }
        Some(self.values().iter().cloned().fold(f64::NAN, max_fold))
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
