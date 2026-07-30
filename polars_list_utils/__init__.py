"""Polars expression plugins for signal processing on List columns.

Four Rust plugins operating elementwise on `List[f64]` columns, plus
one pure-polars helper:

- :func:`apply_interp`: interpolate (x, y) data onto new x coordinates.
- :func:`apply_butterworth`: zero-phase Butterworth filtering.
- :func:`apply_fft`: one-sided FFT with standard windowing and scaling.
- :func:`agg_slices`: aggregate values selected by index-column ranges.
- :func:`agg_lists`: aggregate list columns elementwise over rows.

The plugins accept length-1 literal list columns (e.g. `pl.lit(...)`)
for any input and broadcast them. Null rows and empty lists produce null
output rows. The three transforms also null rows whose lists contain
null elements (a signal with missing samples cannot be transformed);
the two aggregations instead skip missing data polars-style.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_list_utils._internal import __version__ as __version__

IntoExprColumn = str | pl.Expr | pl.Series
Window = Literal["hann", "hanning", "blackman"]
Scaling = Literal["amplitude", "power", "psd"]
Aggregation = Literal["mean", "median", "std", "min", "max", "delta", "count"]

RangeBound = tuple[float, Literal["closed", "open"]]
Range = tuple[float, float] | tuple[RangeBound, RangeBound]
"""An index range: `(lo, hi)` (closed bounds) or
`((lo, "closed"), (hi, "open"))` with explicit boundary modes."""

_LIB = Path(__file__).parent

__all__ = [
    "Aggregation",
    "IntoExprColumn",
    "Range",
    "RangeBound",
    "Scaling",
    "Window",
    "agg_lists",
    "agg_slices",
    "apply_butterworth",
    "apply_fft",
    "apply_interp",
    "polars_func_arg_into_col_expr",
]


def polars_func_arg_into_col_expr(column: IntoExprColumn) -> pl.Expr:
    """Parse an IntoExprColumn type into a Polars Expression (pl.Expr)."""
    if isinstance(column, str):
        return pl.col(column)
    if isinstance(column, pl.Series):
        return pl.lit(column)
    return column


def _plugin(
    function_name: str,
    args: list[IntoExprColumn],
    **kwargs: object,
) -> pl.Expr:
    return register_plugin_function(
        plugin_path=_LIB,
        function_name=function_name,
        args=args,
        kwargs=kwargs or None,
        is_elementwise=True,
    )


def apply_interp(
    x_column: IntoExprColumn,
    y_column: IntoExprColumn,
    xp_column: IntoExprColumn,
) -> pl.Expr:
    """Interpolate `(x, y)` data onto the `xp` coordinates.

    Behaves like `numpy.interp`: linear interpolation, with `xp`
    values outside the data range clamped to the first/last `y` value.
    `x` must be sorted in increasing order.

    Returns a `List[f64]` column of interpolated y values, one per
    `xp` coordinate. Raises if x and y lists differ in length.
    """
    return _plugin(
        "apply_interp",
        [x_column, y_column, xp_column],
    )


def apply_butterworth(
    signal_column: IntoExprColumn,
    *,
    sample_rate: float,
    min_freq: float | None = None,
    max_freq: float | None = None,
    filter_order: int = 4,
) -> pl.Expr:
    """Apply a zero-phase Butterworth filter to a list column of signals.

    `min_freq` sets a highpass cutoff, `max_freq` a lowpass cutoff;
    both together form a bandpass. With neither, the input passes
    through unchanged. Cutoffs must lie within `(0, sample_rate / 2)`.

    Filtering is bidirectional (like `scipy.signal.filtfilt`): zero
    phase lag, squared magnitude response, and the cutoff sits at -6 dB
    rather than -3 dB. Bandpass doubles the design order, as in
    `scipy.signal.butter`. Rows with no more than 3x the effective
    order samples (the reflection padding) yield null.
    """
    return _plugin(
        "apply_butterworth",
        [signal_column],
        sample_rate=sample_rate,
        min_freq=min_freq,
        max_freq=max_freq,
        filter_order=filter_order,
    )


def apply_fft(
    signal_column: IntoExprColumn,
    *,
    sample_rate: float,
    window: Window | None = None,
    scaling: Scaling | None = None,
) -> pl.Expr:
    """Apply a one-sided FFT to a list column of time-domain signals.

    Emits `N/2 + 1` values from DC to Nyquist, so the frequency axis is
    `[i * sample_rate / N for i in range(N // 2 + 1)]`. Signal lengths
    must be powers of two; rows violating that (or containing non-finite
    values) yield null.

    Scaling conventions match scipy for a tone of peak amplitude A:

    - `"amplitude"`: peak-amplitude spectrum, the tone reads A.
    - `"power"`: power spectrum, the tone reads its mean-square A²/2
      (`scipy.signal.periodogram(scaling="spectrum")`).
    - `"psd"`: power spectral density in unit²/Hz; integrates to the
      signal's mean-square power (`scaling="density"`).
    - `None`: raw FFT magnitudes.
    """
    return _plugin(
        "apply_fft",
        [signal_column],
        sample_rate=sample_rate,
        window=window,
        scaling=scaling,
    )


def agg_slices(
    value_column: IntoExprColumn,
    index_column: IntoExprColumn,
    *,
    aggregation: Aggregation,
    slices_include: list[Range] | None = None,
    slices_exclude: list[Range] | None = None,
) -> pl.Expr:
    """Aggregate values whose paired index falls within the given ranges.

    Selects the elements of `value_column` whose corresponding element
    in `index_column` lies in any `slices_include` range and no
    `slices_exclude` range, then aggregates them into a `Float64`.
    Omitting `slices_include` includes everything.

    Missing data follows the polars convention, so results match
    polars' vertical aggregations over the same values: null elements
    are skipped (pairwise with their index), while NaN is a legitimate
    float value — it poisons `mean`/`std`, is skipped by
    `min`/`max` unless all values are NaN, and sorts as the largest
    value for `median`. An empty selection yields null (`count`: 0).
    NaN indices never match any range; mismatched list lengths raise.

    `std` is the sample standard deviation (ddof=1, polars' default)
    and yields null for selections with fewer than two values.
    """
    return _plugin(
        "agg_slices",
        [value_column, index_column],
        aggregation=aggregation,
        slices_include=slices_include,
        slices_exclude=slices_exclude,
    )


_AGGS: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    "mean": pl.Expr.mean,
    "median": pl.Expr.median,
    "std": pl.Expr.std,
    "min": pl.Expr.min,
    "max": pl.Expr.max,
    "delta": lambda e: e.max() - e.min(),
    "count": lambda e: e.count().cast(pl.Float64),
}


def agg_lists(
    list_column: IntoExprColumn,
    *,
    list_length: int,
    aggregation: Aggregation,
) -> pl.Expr:
    """Aggregate a list-type column elementwise over rows.

    For use in a group_by/aggregation context: element n of the result
    is `aggregation` applied vertically to element n of every list in
    the group, giving one `List[f64]` of length `list_length` per
    group. Lists shorter than `list_length` contribute nulls at the
    missing positions.

    Implemented in pure polars (no plugin), so missing data follows the
    polars convention exactly as in :func:`agg_slices`: null elements
    are skipped (`count` counts non-null values), NaN propagates
    per-kernel, and `std` is the sample standard deviation (ddof=1).
    """
    if list_length < 1:
        raise ValueError(f"list_length must be at least 1, got {list_length}")
    agg = _AGGS[aggregation]
    return pl.concat_list(
        agg(
            polars_func_arg_into_col_expr(list_column)
            .list.slice(offset=n, length=1)
            .list.first()
        )
        for n in range(list_length)
    )
