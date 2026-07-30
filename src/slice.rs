use serde::Deserialize;

/// Specifies whether a range boundary is inclusive or exclusive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Bound {
    /// Boundary value is included in the range.
    #[default]
    Closed,
    /// Boundary value is excluded from the range.
    Open,
}

/// A range with lower and upper values and their boundary modes.
///
/// Deserializes from `((lo, bound), (hi, bound))` or the `(lo, hi)`
/// shorthand (closed bounds on both ends).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[serde(from = "RangeRepr")]
pub(crate) struct Range {
    lower: (f64, Bound),
    upper: (f64, Bound),
}

#[derive(Deserialize)]
#[serde(untagged)]
enum RangeRepr {
    Explicit(((f64, Bound), (f64, Bound))),
    Simple((f64, f64)),
}

impl From<RangeRepr> for Range {
    fn from(repr: RangeRepr) -> Self {
        match repr {
            RangeRepr::Explicit((lower, upper)) => Self { lower, upper },
            RangeRepr::Simple((lo, hi)) => Self {
                lower: (lo, Bound::Closed),
                upper: (hi, Bound::Closed),
            },
        }
    }
}

impl Range {
    /// Create a range that includes everything.
    fn full() -> Self {
        Self {
            lower: (f64::NEG_INFINITY, Bound::Closed),
            upper: (f64::INFINITY, Bound::Closed),
        }
    }

    /// Check if a value is within this range given the boundary modes.
    fn contains(
        &self,
        value: f64,
    ) -> bool {
        let left_ok = match self.lower.1 {
            Bound::Closed => value >= self.lower.0,
            Bound::Open => value > self.lower.0,
        };
        let right_ok = match self.upper.1 {
            Bound::Closed => value <= self.upper.0,
            Bound::Open => value < self.upper.0,
        };
        left_ok && right_ok
    }
}

/// A set of index ranges to include, minus a set of ranges to exclude.
#[derive(Debug, Clone)]
pub(crate) struct Slice {
    included: Vec<Range>,
    excluded: Vec<Range>,
}

impl Slice {
    /// Create a new empty Slice.
    pub(crate) fn empty() -> Self {
        Self {
            included: vec![],
            excluded: vec![],
        }
    }

    /// Create a new Slice that includes the full range.
    pub(crate) fn full() -> Self {
        Self {
            included: vec![Range::full()],
            excluded: vec![],
        }
    }

    /// Add a range to include.
    pub(crate) fn include(
        mut self,
        range: Range,
    ) -> Self {
        self.included.push(range);
        self
    }

    /// Add a range to exclude.
    pub(crate) fn exclude(
        mut self,
        range: Range,
    ) -> Self {
        self.excluded.push(range);
        self
    }

    /// Check whether an index satisfies the include/exclude conditions.
    pub(crate) fn contains(
        &self,
        index: f64,
    ) -> bool {
        self.included.iter().any(|r| r.contains(index))
            && !self.excluded.iter().any(|r| r.contains(index))
    }
}
