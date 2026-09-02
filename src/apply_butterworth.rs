use crate::filter::Filter;
use crate::util::{Container, apply_list_transform, require_float_inner};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

#[derive(Deserialize)]
struct ButterworthKwargs {
    sample_rate: f64,
    min_freq: Option<f64>,
    max_freq: Option<f64>,
    filter_order: Option<usize>,
}

impl ButterworthKwargs {
    fn filter(&self) -> Option<Filter> {
        match (self.min_freq, self.max_freq) {
            (Some(lo), Some(hi)) => Some(Filter::Bandpass(lo, hi)),
            (Some(lo), None) => Some(Filter::Highpass(lo)),
            (None, Some(hi)) => Some(Filter::Lowpass(hi)),
            (None, None) => None,
        }
    }

    fn order(&self) -> usize {
        self.filter_order.unwrap_or(4)
    }
}

/// Length-preserving, so the output keeps the signal's container and
/// inner dtype. The kwargs are available here, which lets an `Array`
/// too short for the reflection padding raise at plan time — the width
/// is in the schema, and a fixed width means every row would fail.
fn butterworth_output(
    input_fields: &[Field],
    kwargs: ButterworthKwargs,
) -> PolarsResult<Field> {
    let (container, inner) =
        require_float_inner(input_fields[0].dtype(), "apply_butterworth: signal_column")?;
    polars_ensure!(
        kwargs.sample_rate > 0.0,
        ComputeError: "apply_butterworth: sample_rate must be positive, got {}",
        kwargs.sample_rate
    );
    if let (Container::Array(w), Some(filter)) = (container, kwargs.filter()) {
        let min_samples = filter.min_samples(kwargs.order());
        polars_ensure!(
            w >= min_samples,
            ComputeError:
            "apply_butterworth: signal_column has Array width {w} but the order-{} \
            filter needs at least {min_samples} samples (reflection padding)",
            kwargs.order()
        );
    }
    Ok(Field::new(PlSmallStr::from(""), container.dtype(inner)))
}

/// Apply a zero-phase Butterworth filter to a list column.
///
/// # Arguments (via kwargs)
/// - `sample_rate`: Sampling rate in Hz
/// - `min_freq`: Highpass cutoff frequency (Hz)
/// - `max_freq`: Lowpass cutoff frequency (Hz)
/// - `filter_order`: Filter order (default 4)
///
/// If both cutoffs are specified, a bandpass filter is applied; with one,
/// a highpass or lowpass; with neither, the input passes through (the
/// family entry contract still applies: null-element rows null out, and
/// an empty row raises — a bypass that skipped it would make row
/// semantics depend on parameter values).
///
/// Filtering is bidirectional (filtfilt): zero phase lag, squared
/// magnitude response, cutoff at -6 dB. Bandpass doubles the design
/// order, as in scipy.signal.butter. A row with no more than 3x the
/// effective order samples (the reflection padding) raises — wrong shape
/// is structural, not data. Non-finite samples are legitimate floats and
/// contaminate the filtered output rather than nulling it.
#[polars_expr(output_type_func_with_kwargs=butterworth_output)]
fn apply_butterworth(
    inputs: &[Series],
    kwargs: ButterworthKwargs,
) -> PolarsResult<Series> {
    polars_ensure!(
        kwargs.sample_rate > 0.0,
        ComputeError: "apply_butterworth: sample_rate must be positive, got {}",
        kwargs.sample_rate
    );
    let (out, inner) =
        require_float_inner(inputs[0].dtype(), "apply_butterworth: signal_column")?;
    let Some(filter) = kwargs.filter() else {
        return apply_list_transform(
            inputs,
            "apply_butterworth",
            &[false],
            out,
            &inner,
            |cols| Ok(Some(cols[0].to_vec())),
        );
    };
    let order = kwargs.order();
    let built = filter.build(kwargs.sample_rate, order)?;
    let min_samples = filter.min_samples(order);

    apply_list_transform(
        inputs,
        "apply_butterworth",
        &[false],
        out,
        &inner,
        move |cols| {
            polars_ensure!(
                cols[0].len() >= min_samples,
                ComputeError:
                "apply_butterworth: a row's signal has {} samples but the order-{order} \
                filter needs at least {min_samples} (reflection padding; pre-filter \
                with list.len() to drop such rows)",
                cols[0].len()
            );
            // An internal filter failure is a bug surfacing, not
            // missing data: raise rather than null.
            built.bidirectional(&cols[0].to_vec()).map(Some).map_err(
                |e| polars_err!(ComputeError: "apply_butterworth: filter failed: {e:?}"),
            )
        },
    )
}
