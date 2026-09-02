"""Polars expression plugins for signal processing on list columns.

Six elementwise Rust plugins, plus :func:`agg_lists`, which composes
polars' own vertical aggregations (with a small pass-through plugin to
prepare its input). Value-like inputs take `Float32` or `Float64`
inners — anything else, integers included, raises at plan time naming
the parameter — and `Float32` is preserved: kernels compute in `f64`
and round once at the output boundary. The one exception to dtype
preservation is `count`, which emits `UInt32`, polars' count dtype.
Gates and the Boolean operands of `zip_binary` take Boolean inners:

- :func:`apply_interp`: interpolate (x, y) data onto new x coordinates.
- :func:`apply_butterworth`: zero-phase Butterworth filtering.
- :func:`apply_fft`: one-sided FFT with standard windowing and scaling.
- :func:`agg_slices`: aggregate values selected by index-column ranges.
- :func:`agg_lists`: aggregate list columns elementwise over rows.
- :func:`zip_binary`: element-wise binary ops between two list columns.
- :func:`cum_agg_runs`: cumulative aggregation within gated runs.

Both `List` and `Array` columns are accepted everywhere, and a
length-preserving function returns the container it was given —
`apply_fft` narrows an `Array(w)` to `Array(w // 2 + 1)`, and
`apply_interp` follows its `xp` argument, since that is what decides the
output's shape. `zip_binary` returns an `Array` only when both operands
are `Array`s, since both are operands; `cum_agg_runs` follows its value
column alone, its gate being a mask rather than an operand. Paired
inputs that must agree element-for-element have their widths checked
before execution when both are `Array`s, and per row otherwise. Two
exceptions to container propagation: :func:`agg_slices` reduces to a
scalar, and :func:`agg_lists` always returns a `List`, being an
expression composition that never sees its input's schema. Zero-width
`Array`s are rejected — polars panics when slicing them — while
zero-length `List`s are fine.

The plugins accept length-1 literal list columns (e.g. `pl.lit(...)`)
for any input and broadcast them. Null rows produce null output rows.

Data conditions follow three deliberate family regimes. A signal is
atomic, so for the transforms (`apply_interp`, `apply_butterworth`,
`apply_fft`) the rule is three lines: **missing data nulls** — a null
row or any null element yields a null row, a signal with missing
samples being untransformable; **invalid values propagate** — NaN and
±inf are legitimate floats that flow through the arithmetic and stay
visible in the output; **wrong shape raises** — a length the transform
cannot process, including the empty list, is structural, not data.
The aggregations (`agg_slices`, `agg_lists`) instead skip missing data
polars-style, and treat an empty list as an empty selection. And
`zip_binary`/`cum_agg_runs` mirror polars' scalar and `cum_*`
operations per element: nulls propagate per those ops' rules, and
empty lists are valid (empty in, empty out — total functions over zero
elements).
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
Scaling = Literal["amplitude", "amplitude_squared", "spectrum", "density"]
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

    Empty splits precisely: an empty `x`/`y` is no interpolant and
    raises, while an empty `xp` is zero query points — a valid question
    whose answer is an empty list. The output's container follows `xp`
    and its inner dtype follows `y` (values determine the value dtype).

    With `strict=True` (the default) a row whose `x` values descend
    raises; duplicates stay legal, as in numpy. Pass `strict=False` for
    numpy's silent behaviour, where unsorted `x` interpolates garbage
    without complaint. A NaN in `x` is a legitimate float either way
    and propagates into the output rather than raising — the check
    exists for the failure you cannot see, not the one you can.

    Returns one interpolated y value per `xp` coordinate, in the
    container and inner dtype the paragraph above describes. Raises if
    x and y lists differ in length.
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
    `scipy.signal.butter`. A row with no more than 3x the effective
    order samples (the reflection padding) raises — wrong shape is
    structural, not data; for an `Array` the width is in the schema and
    the raise lands at plan time, and for mixed-length `List` data the
    escape hatch is pre-filtering with `list.len()` before this
    transform (in a lazy query the optimizer pushes a later filter
    below it, dropping offending rows before validation rather than
    raising). Non-finite samples
    propagate and contaminate the filtered output rather than nulling
    it.
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
    must be powers of two of at least 2 — a violation raises (for an
    `Array`, already at plan time, the width being in the schema; for
    mixed-length `List` data, pre-filter with `list.len()` *before*
    this transform — in a lazy query, a filter written after it is
    pushed below it by the optimizer, so offending rows are dropped
    before validation rather than raising). Non-finite
    samples are legitimate floats and propagate: every bin sums all
    samples, so one NaN yields an all-NaN spectrum rather than a null
    row — invalid data stays visibly invalid instead of becoming
    missing. Guard downstream thresholds with `is_finite()`.

    Scaling names are scipy's, for a tone of peak amplitude A:

    - `"amplitude"`: peak-amplitude spectrum, the tone reads A.
    - `"amplitude_squared"`: `amplitude`² per bin, the tone reads A².
      A convenience with no external anchor — elementwise squaring of a
      list column is exactly the boilerplate this library removes.
    - `"spectrum"`: power spectrum, the tone reads its mean-square A²/2
      (`scipy.signal.periodogram(scaling="spectrum")`).
    - `"density"`: power spectral density in unit²/Hz; integrates to
      the signal's mean-square power (`scaling="density"`).
    - `None`: raw FFT magnitudes of the windowed signal.

    The one-sided doubling happens in each scaling's own domain and
    skips DC and Nyquist, so `spectrum != amplitude²/2` at exactly those
    two bins — matching scipy. Windows are periodic (sym=False), as in
    `periodogram`.
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
    `slices_exclude` range, then aggregates them into a scalar that
    keeps the value column's inner dtype (`count` alone emits `UInt32`,
    polars' count dtype). Omitting `slices_include` includes everything.

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

    An empty selection — no index fell in range, or the lists were
    themselves empty, the limiting case of the same condition — yields
    null, except `count`, which yields 0. `empty` decides the one
    aggregation where the two conventions disagree:

    - `"null"` (default): summing nothing is unknown, not zero. An
      empty selection usually means a misconfigured range, and a real
      `0.0` disguises that as a measurement — the dangerous direction
      for a feature feeding a threshold.
    - `"zero"`: polars' convention, the identity element of addition,
      matching `explode().group_by().agg(col.filter(...).sum())`.

    `empty` governs the empty *selection* only. A null row stays null
    under both settings for every aggregation including `count` — so
    `empty="zero"` is not the same as `.fill_null(0.0)`, which cannot
    tell a missing row from an empty selection and would launder the
    former into a real zero.

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
    "count": pl.Expr.count,  # UInt32, polars' count dtype
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
    the group, giving one `List` of length `list_length` per group.
    The result is always a `List`, even for `Array` input: this is an
    expression composition rather than a plugin, because it reduces many
    rows to one, and an expression is built before any schema is
    resolved — so the container to return is not knowable here.

    `List` rows shorter than `list_length` contribute nulls at the
    missing positions. An `Array` input must instead match `list_length`
    exactly, since its width is fixed for every row and a mismatch is
    therefore a configuration error rather than data; it raises, under
    `strict` or not.

    With `strict=True` (the default), `List` rows carrying non-null data
    past `list_length` raise. Pass `strict=False` to keep the silent
    truncation, which some callers want as a deliberate window. A
    null-padded tail is never a violation: every aggregation skips
    nulls, so truncating it loses nothing.

    The aggregations are polars' own, so missing data follows the polars
    convention exactly as in :func:`agg_slices`: null elements are
    skipped (`count` counts non-null values and emits `UInt32`, polars'
    count dtype), NaN propagates per-kernel, and `std` is the sample
    standard deviation (ddof=1). `empty` decides
    what an all-null position sums to — null by default, `0.0` under
    `empty="zero"` — exactly as in :func:`agg_slices`.

    A small pass-through plugin runs first, to normalize the container
    and carry the checks above; polars' `.list` namespace rejects
    `Array` outright, and an expression cannot raise from inside itself.
    """
    if list_length < 1:
        raise ValueError(f"list_length must be at least 1, got {list_length}")
    _check_choice(aggregation, _AGGREGATIONS, "aggregation", "agg_lists")
    _check_choice(empty, _EMPTY, "empty", "agg_lists")
    # The pass-through plugin normalizes Array to List (the `.list`
    # namespace rejects Array, and an expression cannot branch on a dtype
    # it has not resolved yet) and, under strict, raises on rows that
    # would lose data — which pure polars cannot do from inside an
    # expression.
    col = _plugin(
        "prepare_agg_lists",
        [polars_func_arg_into_col_expr(list_column)],
        list_length=list_length,
        strict=strict,
    )
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
