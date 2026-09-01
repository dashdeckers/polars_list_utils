use crate::statistic::{Aggregation, Statistic, max_fold, midpoint, min_fold};
use crate::util::{
    TypedListInput, broadcast_len, build_float_list, build_u32_list, tot_cmp,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cmp::Ordering;

/// What non-`True`-gated positions emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Outside {
    Null,
    Zero,
}

#[derive(Deserialize)]
struct CumAggRunsKwargs {
    aggregation: Aggregation,
    outside: Outside,
}

/// Running state over one gated run's non-null values.
///
/// Every emitted position must equal the vertical aggregation over the
/// run-prefix's non-null values (the prefix rule), so `std` and
/// `median` delegate to the very kernels `agg_slices` uses rather than
/// keeping independent accumulators: one input, one answer, everywhere
/// in the library. That costs a retained buffer for those two
/// aggregations; the rest need O(1) state.
///
/// `sum`/`mean` let NaN poison the running sum, `min`/`max` skip NaN
/// (all-NaN prefixes read NaN) and keep the first of equal values, and
/// `count` counts every non-null value, NaN included.
struct RunState {
    n: u64,
    sum: f64,
    min: f64,
    max: f64,
    /// The run's values in input order, kept for `std` only.
    values: Vec<f64>,
    /// The run's values in stable sorted order, kept for `median` only.
    sorted: Vec<f64>,
}

impl RunState {
    fn new() -> Self {
        Self {
            n: 0,
            sum: 0.0,
            min: f64::NAN,
            max: f64::NAN,
            values: Vec::new(),
            sorted: Vec::new(),
        }
    }

    fn update(
        &mut self,
        agg: Aggregation,
        x: f64,
    ) {
        self.n += 1;
        self.sum += x;
        self.min = min_fold(self.min, x);
        self.max = max_fold(self.max, x);
        match agg {
            Aggregation::Std => self.values.push(x),
            Aggregation::Median => {
                // Upper bound: insert *after* equal elements, so the
                // buffer holds what a stable sort would produce and a
                // signed-zero tie keeps its input order, as in polars.
                let pos = self
                    .sorted
                    .partition_point(|&v| tot_cmp(v, x) != Ordering::Greater);
                self.sorted.insert(pos, x);
            }
            _ => {}
        }
    }

    /// The aggregation over the prefix; only called with `n >= 1`.
    fn emit(
        &self,
        agg: Aggregation,
    ) -> Option<f64> {
        match agg {
            Aggregation::Sum => Some(self.sum),
            Aggregation::Count => Some(self.n as f64),
            Aggregation::Mean => Some(self.sum / self.n as f64),
            Aggregation::Min => Some(self.min),
            Aggregation::Max => Some(self.max),
            Aggregation::Delta => Some(self.max - self.min),
            // Sample std (ddof=1): null below two values, as in polars.
            Aggregation::Std => self.values.std(),
            Aggregation::Median => {
                let mid = self.sorted.len() / 2;
                Some(if self.sorted.len().is_multiple_of(2) {
                    midpoint(self.sorted[mid - 1], self.sorted[mid])
                } else {
                    self.sorted[mid]
                })
            }
        }
    }
}

/// Scan one row: a run is a maximal region of constant gate value
/// (`True`, `False`, and null are distinct, so null gates break runs).
/// Only `True`-gated positions accumulate; the rest emit the `outside`
/// fill. Null values leave the running state unchanged and emit null,
/// except `count`, which emits the unchanged running count (as native
/// `cum_count` does).
fn scan_row(
    values: &[Option<f64>],
    gates: &[Option<bool>],
    agg: Aggregation,
    outside: Outside,
) -> PolarsResult<Vec<Option<f64>>> {
    // No row index: plugins are handed one chunk at a time, so any
    // position we could name here is chunk-local and would point at a
    // different, innocent row of the frame.
    polars_ensure!(
        values.len() == gates.len(),
        ComputeError:
        "cum_agg_runs: a row's value and gate lists differ in length ({} vs {})",
        values.len(), gates.len()
    );

    let outside_fill = match outside {
        Outside::Null => None,
        Outside::Zero => Some(0.0),
    };

    let mut out = Vec::with_capacity(values.len());
    let mut state = RunState::new();
    let mut prev_gate: Option<Option<bool>> = None;
    for (&value, &gate) in values.iter().zip(gates) {
        if prev_gate != Some(gate) {
            state = RunState::new();
        }
        prev_gate = Some(gate);

        if gate != Some(true) {
            out.push(outside_fill);
            continue;
        }
        match value {
            None => out.push(match agg {
                Aggregation::Count => Some(state.n as f64),
                _ => None,
            }),
            Some(x) => {
                state.update(agg, x);
                out.push(state.emit(agg));
            }
        }
    }
    Ok(out)
}

fn cum_agg_runs_output(
    input_fields: &[Field],
    kwargs: CumAggRunsKwargs,
) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 2,
        ComputeError: "cum_agg_runs expects exactly 2 inputs, got {}",
        input_fields.len()
    );
    let value_inner = match input_fields[0].dtype() {
        DataType::List(inner)
            if matches!(**inner, DataType::Float32 | DataType::Float64) =>
        {
            *inner.clone()
        }
        dt => polars_bail!(ComputeError:
            "cum_agg_runs: value_column must be a List column with Float32 or Float64 \
            inner dtype, got {dt}"),
    };
    polars_ensure!(
        matches!(input_fields[1].dtype(), DataType::List(inner) if **inner == DataType::Boolean),
        ComputeError: "cum_agg_runs: gate_column must be a List column with Boolean inner \
        dtype, got {}",
        input_fields[1].dtype()
    );
    // count emits UInt32, polars' count dtype; everything else keeps
    // the value column's inner dtype.
    let inner = match kwargs.aggregation {
        Aggregation::Count => DataType::UInt32,
        _ => value_inner,
    };
    Ok(Field::new(
        PlSmallStr::from(""),
        DataType::List(Box::new(inner)),
    ))
}

/// Cumulative aggregation within gated runs of a list column.
///
/// A null row in either input yields a null row; empty lists scan to
/// empty lists; mismatched value/gate lengths raise.
#[polars_expr(output_type_func_with_kwargs=cum_agg_runs_output)]
fn cum_agg_runs(
    inputs: &[Series],
    kwargs: CumAggRunsKwargs,
) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "cum_agg_runs expects exactly 2 inputs, got {}",
        inputs.len()
    );
    let values = TypedListInput::<f64>::float(&inputs[0], "cum_agg_runs: value_column")?;
    let gates = TypedListInput::<bool>::boolean(&inputs[1], "cum_agg_runs: gate_column")?;
    let len = broadcast_len(&[values.n_rows(), gates.n_rows()])?;

    let mut rows: Vec<Option<Vec<Option<f64>>>> = Vec::with_capacity(len);
    for i in 0..len {
        rows.push(match (values.row(i), gates.row(i)) {
            (Some(v), Some(g)) => {
                Some(scan_row(v, g, kwargs.aggregation, kwargs.outside)?)
            }
            _ => None,
        });
    }

    if kwargs.aggregation == Aggregation::Count {
        // Counts are exact integers well below 2^53, so the f64 detour
        // is lossless.
        let rows = rows
            .into_iter()
            .map(|row| {
                row.map(|vals| {
                    vals.into_iter()
                        .map(|v| v.map(|x| x as u32))
                        .collect::<Vec<Option<u32>>>()
                })
            })
            .collect();
        Ok(build_u32_list(rows))
    } else {
        build_float_list(rows, values.inner_dtype())
    }
}
