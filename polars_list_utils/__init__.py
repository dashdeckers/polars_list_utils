"""Polars expression plugins for signal processing on List columns.

Six Rust plugins operating elementwise on List columns, plus one
pure-polars helper. The four original functions take `List[f64]`
(anything numeric is cast to it); :func:`zip_binary` and
:func:`cum_agg_runs` instead take float or Boolean inners as their
operands require, and preserve `Float32`:

- :func:`apply_interp`: interpolate (x, y) data onto new x coordinates.
- :func:`apply_butterworth`: zero-phase Butterworth filtering.
- :func:`apply_fft`: one-sided FFT with standard windowing and scaling.
- :func:`agg_slices`: aggregate values selected by index-column ranges.
- :func:`agg_lists`: aggregate list columns elementwise over rows.
- :func:`zip_binary`: element-wise binary ops between two list columns.
- :func:`cum_agg_runs`: cumulative aggregation within gated runs.

The plugins accept length-1 literal list columns (e.g. `pl.lit(...)`)
for any input and broadcast them. Null rows produce null output rows.

Missing data follows three deliberate family regimes:

- Transforms (`apply_interp`, `apply_butterworth`, `apply_fft`) null
  out rows whose lists are empty or contain null elements (a signal
  with missing samples cannot be transformed).
- Aggregations (`agg_slices`, `agg_lists`) skip missing data
  polars-style; NaN is a legitimate float and flows through.
- `zip_binary` and `cum_agg_runs` mirror polars' scalar and `cum_*`
  operations per element: nulls propagate per those ops' rules, and
  empty lists are valid (empty in, empty out).
"""

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_list_utils._internal import __version__ as __version__

IntoExprColumn = str | pl.Expr | pl.Series
Window = Literal["hann", "hanning", "blackman"]
Scaling = Literal["amplitude", "power", "psd"]
Aggregation = Literal["sum", "mean", "median", "std", "min", "max", "delta", "count"]
Empty = Literal["null", "zero"]

_AGGREGATIONS = ("sum", "mean", "median", "std", "min", "max", "delta", "count")
_EMPTY = ("null", "zero")
BinaryOp = Literal[
    "add", "sub", "mul", "div", "and", "or", "gt", "ge", "lt", "le", "eq", "ne"
]

RangeBound = tuple[float, Literal["closed", "open"]]
Range = tuple[float, float] | tuple[RangeBound, RangeBound]
"""An index range: `(lo, hi)` (closed bounds) or
`((lo, "closed"), (hi, "open"))` with explicit boundary modes."""

_LIB = Path(__file__).parent

__all__ = [
    "Aggregation",
    "BinaryOp",
    "Empty",
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
    "cum_agg_runs",
    "polars_func_arg_into_col_expr",
    "zip_binary",
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
    *,
    strict: bool = True,
) -> pl.Expr:
    """Interpolate `(x, y)` data onto the `xp` coordinates.

    Behaves like `numpy.interp`: linear interpolation, with `xp`
    values outside the data range clamped to the first/last `y` value.
    `x` must be sorted in increasing order.

    With `strict=True` (the default) a row whose `x` values descend
    raises; duplicates stay legal, as in numpy. Pass `strict=False` for
    numpy's silent behaviour, where unsorted `x` interpolates garbage
    without complaint. A NaN in `x` is a legitimate float either way
    and propagates into the output rather than raising — the check
    exists for the failure you cannot see, not the one you can.

    Returns a `List[f64]` column of interpolated y values, one per
    `xp` coordinate. Raises if x and y lists differ in length.
    """
    return _plugin(
        "apply_interp",
        [x_column, y_column, xp_column],
        strict=strict,
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


def _check_choice(
    value: object,
    allowed: tuple[str, ...],
    param: str,
    func: str,
) -> None:
    """Reject an out-of-vocabulary keyword at expression construction.

    Both aggregating functions validate here rather than leaving it to
    the plugin, so a typo fails the same way in each. It matters most
    for `empty`, whose two settings differ in exactly the direction
    this library cares about: silently falling through to `"zero"`
    would turn "no data" into a real 0.0, the failure the flag exists
    to prevent.
    """
    if value not in allowed:
        raise ValueError(
            f"{func}: {param} must be one of {', '.join(map(repr, allowed))}, "
            f"got {value!r}"
        )


def _normalize_ranges(
    ranges: list[Range] | None, param: str, strict: bool
) -> list[Range] | None:
    """Rebuild `ranges` into fresh plain-float tuples, validating under strict.

    The rebuild is not cosmetic. The plugin receives kwargs as a pickle,
    and its decoder cannot resolve a memoized *container* back-reference:
    any range object (or bound tuple, or the range list itself) that
    appears twice — as CPython's constant folding arranges for a literal
    like `((1.0, "closed"), (1.0, "closed"))` — would otherwise fail the
    query with "recursive structure found". Fresh objects are never
    pickled twice. Coercing bounds with `float()` likewise keeps numpy
    scalars, which the decoder cannot resolve either, off the wire.
    """
    if ranges is None:
        return None

    def bound(b: Any) -> tuple[float, str] | float:
        # Explicit ((value, mode)) form, as a tuple or (from JSON) a list.
        if isinstance(b, (tuple, list)):
            value, mode = b
            return (float(value), str(mode))
        return float(b)

    out: list[Range] = []
    for r in ranges:
        lo, hi = (bound(b) for b in r)
        lo_value = lo[0] if isinstance(lo, tuple) else lo
        hi_value = hi[0] if isinstance(hi, tuple) else hi
        if strict and (
            math.isnan(lo_value) or math.isnan(hi_value) or lo_value > hi_value
        ):
            raise ValueError(
                f"agg_slices: invalid range {r!r} in {param} (strict=True): "
                "bounds must be non-NaN with lo <= hi"
            )
        out.append((lo, hi))  # ty: ignore[invalid-argument-type]
    return out


def agg_slices(
    value_column: IntoExprColumn,
    index_column: IntoExprColumn,
    *,
    aggregation: Aggregation,
    slices_include: list[Range] | None = None,
    slices_exclude: list[Range] | None = None,
    empty: Empty = "null",
    strict: bool = True,
) -> pl.Expr:
    """Aggregate values whose paired index falls within the given ranges.

    Selects the elements of `value_column` whose corresponding element
    in `index_column` lies in any `slices_include` range and no
    `slices_exclude` range, then aggregates them into a `Float64`.
    Omitting `slices_include` includes everything.

    Missing data follows the polars convention, so results follow
    polars' vertical aggregations over the same values (semantically —
    floating-point accumulation order differs, so agreement is to
    rounding, not bit-for-bit): null elements
    are skipped (pairwise with their index), while NaN is a legitimate
    float value — it poisons `sum`/`mean`/`std`, is skipped by
    `min`/`max` unless all values are NaN, and sorts as the largest
    value for `median`. NaN indices never match any range; mismatched
    list lengths raise.

    `std` is the sample standard deviation (ddof=1, polars' default)
    and yields null for selections with fewer than two values.

    An empty selection — the lists held elements, but no index fell in
    range — yields null, except `count`, which yields 0. `empty`
    decides the one aggregation where the two conventions disagree:

    - `"null"` (default): summing nothing is unknown, not zero. An
      empty selection usually means a misconfigured range, and a real
      `0.0` disguises that as a measurement — the dangerous direction
      for a feature feeding a threshold.
    - `"zero"`: polars' convention, the identity element of addition,
      matching `explode().group_by().agg(col.filter(...).sum())`.

    `empty` governs the empty *selection* only. A null row, and a row
    whose lists are themselves empty, stay null under both settings for
    every aggregation including `count` — so `empty="zero"` is not the
    same as `.fill_null(0.0)`, which cannot tell a missing row from an
    empty selection and would launder the former into a real zero.

    Range bounds accept anything `float()` converts, and the explicit
    form may be given as tuples or lists (so ranges loaded from JSON
    work as-is).

    With `strict=True` (the default), inverted (`lo > hi`) or NaN range
    bounds raise at expression construction. Pass `strict=False` to
    keep them as legitimately empty selections.
    """
    _check_choice(aggregation, _AGGREGATIONS, "aggregation", "agg_slices")
    _check_choice(empty, _EMPTY, "empty", "agg_slices")
    return _plugin(
        "agg_slices",
        [value_column, index_column],
        aggregation=aggregation,
        slices_include=_normalize_ranges(slices_include, "slices_include", strict),
        slices_exclude=_normalize_ranges(slices_exclude, "slices_exclude", strict),
        empty=empty,
    )


_AGGS: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    # polars' own sum of nothing is 0.0; `empty="null"` restores the
    # "no values, no answer" reading by guarding on the non-null count.
    "sum": lambda e: pl.when(e.count() > 0).then(e.sum()),
    "mean": pl.Expr.mean,
    "median": pl.Expr.median,
    "std": pl.Expr.std,
    "min": pl.Expr.min,
    "max": pl.Expr.max,
    "delta": lambda e: e.max() - e.min(),
    "count": lambda e: e.count().cast(pl.Float64),
}

_AGGS_EMPTY_ZERO: dict[str, Callable[[pl.Expr], pl.Expr]] = {
    **_AGGS,
    "sum": pl.Expr.sum,
}


def agg_lists(
    list_column: IntoExprColumn,
    *,
    list_length: int,
    aggregation: Aggregation,
    empty: Empty = "null",
    strict: bool = True,
) -> pl.Expr:
    """Aggregate a list-type column elementwise over rows.

    For use in a group_by/aggregation context: element n of the result
    is `aggregation` applied vertically to element n of every list in
    the group, giving one `List[f64]` of length `list_length` per
    group. Lists shorter than `list_length` contribute nulls at the
    missing positions.

    With `strict=True` (the default), lists longer than `list_length`
    raise. Pass `strict=False` to keep the silent truncation, which
    some callers want as a deliberate window.

    Implemented in pure polars (no plugin), so missing data follows the
    polars convention exactly as in :func:`agg_slices`: null elements
    are skipped (`count` counts non-null values), NaN propagates
    per-kernel, and `std` is the sample standard deviation (ddof=1).
    `empty` decides what an all-null position sums to — null by
    default, `0.0` under `empty="zero"` — exactly as in
    :func:`agg_slices`.
    """
    if list_length < 1:
        raise ValueError(f"list_length must be at least 1, got {list_length}")
    _check_choice(aggregation, _AGGREGATIONS, "aggregation", "agg_lists")
    _check_choice(empty, _EMPTY, "empty", "agg_lists")
    col = polars_func_arg_into_col_expr(list_column)
    if strict:
        # Pure polars cannot raise from inside an expression, so the
        # length check rides along as an elementwise pass-through plugin.
        col = _plugin("check_list_len", [col], list_length=list_length)
    agg = (_AGGS if empty == "null" else _AGGS_EMPTY_ZERO)[aggregation]
    return pl.concat_list(
        agg(col.list.slice(offset=n, length=1).list.first())
        for n in range(list_length)
    )


def zip_binary(
    left_column: IntoExprColumn,
    right_column: IntoExprColumn,
    *,
    op: BinaryOp,
) -> pl.Expr:
    """Apply a binary operation element-wise between two list columns.

    Per element the result is identical to the corresponding scalar
    polars op:

    - Arithmetic (`add`, `sub`, `mul`, `div`) takes float lists; null
      propagates, NaN follows IEEE float arithmetic (`1/0` is inf,
      `0/0` is NaN). The output inner dtype follows polars supertyping:
      `Float32` when both inputs are `Float32`, else `Float64`.
    - `and`/`or` take Boolean lists and follow Kleene logic
      (`False & null = False`, `True | null = True`).
    - Comparisons (`gt`, `ge`, `lt`, `le`, `eq`, `ne`) take float lists
      and emit Boolean, following polars' total order: NaN equals NaN
      and exceeds everything else; comparing against null yields null.

    Integer inner dtypes raise. A null row on either side yields a null
    row; empty lists zip to empty lists; mismatched list lengths raise.
    Length-1 literal lists broadcast (zip against a constant template).
    """
    return _plugin(
        "zip_binary",
        [left_column, right_column],
        op=op,
    )


def cum_agg_runs(
    value_column: IntoExprColumn,
    gate_column: IntoExprColumn,
    *,
    aggregation: Aggregation,
    outside: Literal["null", "zero"] = "null",
) -> pl.Expr:
    """Cumulatively aggregate list values within gated runs.

    A run is a maximal region of constant `gate_column` value (`True`,
    `False`, and null are distinct, so a null gate element breaks
    runs). Only `True`-gated elements accumulate; every other position
    emits the `outside` fill (`"null"`, or `"zero"` for a plain 0 in
    the output dtype). Within a run, each emitted element equals the
    vertical `aggregation` over the run's elements so far — so with an
    all-`True` gate over `Float64` values, `sum`/`min`/`max`/`count`
    track polars' `cum_sum`/`cum_min`/`cum_max`/`cum_count`, with two
    documented exceptions:

    - `cum_min`/`cum_max` fold from ∓`f64::MAX` and never replace that
      seed on a tie, so they emit ±1.7976931348623157e+308 wherever the
      prefix is all-NaN or opens with an infinity of the same sign.
      This function follows the prefix rule and emits the true prefix
      minimum or maximum (NaN, or that infinity).
    - `Float32` values accumulate in `f64` and round once at the output
      boundary, while polars rounds every step, so the two can differ
      in the last bits and around the `Float32` range limit (where this
      function can stay finite as polars overflows to infinity).

    Floating-point accumulation order differs from polars generally, so
    treat the correspondence as one of semantics — missing data, NaN,
    dtype and ddof rules — rather than bit-exact equality.

    Missing data follows the polars `cum_*` convention: a null value
    leaves the running state unchanged and emits null — except `count`,
    which emits the unchanged running count, as `cum_count` does. NaN
    follows the vertical convention from its entry onward within the
    run: it poisons `sum`/`mean`/`std`, is skipped by
    `min`/`max`/`delta`, sorts largest for `median`, and is counted by
    `count`. `std` (ddof=1) emits null until a run prefix holds two
    non-null values.

    `value_column` takes float lists (`Float32` is preserved in the
    output; integers raise) and `gate_column` Boolean lists. `count`
    emits `UInt32`, polars' count dtype. A null row in either input
    yields a null row; empty lists produce empty lists; mismatched
    value/gate lengths raise. Length-1 literal lists broadcast.
    """
    return _plugin(
        "cum_agg_runs",
        [value_column, gate_column],
        aggregation=aggregation,
        outside=outside,
    )
