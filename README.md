# Polars List Utils (`polist`)

[`polist`](https://github.com/dashdeckers/polars_list_utils) is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames, especially for signal processing and feature extraction.

So far these utilities comprise those that I found to be missing or lacking from
the List namespace within the Polars library while I was working on a project at
work that required extensive handling of signal data which I was storing in Polars
DataFrames.

By providing these utilities as a Polars plugin, and thus not having to leave
the Polars DataFrame for these operations, I was able to significantly speed up
the processing of my data by benefiting from Polars query optimization and
parallelization. So while the operations themselves are not necessarily faster
than their Numpy counterparts (although they might be in some cases), the
integration with Polars gave my larger processing pipeline a significant speed
boost.

## A disclaimer about authorship

This library started life as a hand-crafted solution to a problem at work. The time
pressures of life and work eventually led me to succumb to taking shortcuts and let AI
take the wheel. This repo is therefore no longer "artisanal": it is AI-generated, reviewed
to the best of my time and abilities, and fully dog-fooded by me at work.

## Features

- `polist.apply_interp`
    - Interpolates a new List-type column from 3 specified List-type columns.
    - Behaviour as expected from the `numpy.interp` function, but for Polars DataFrames.
    - Supply the `x_column`, `y_column`, and `xp_column` columns to obtain the interpolated y-values.

- `polist.apply_butterworth`
    - Applies a Butterworth filter (low-pass, high-pass, band-pass) to a List-type column of signal data.
    - The filter is applied bidirectionally (like `scipy.signal.filtfilt`), so there is no phase lag,
      but the magnitude response is squared: the cutoff sits at -6dB instead of -3dB.
    - A band-pass doubles the design order, as in `scipy.signal.butter`.

- `polist.apply_fft`
    - Applies a (real) Fast Fourier Transform (FFT) to a List-type column of signal data,
      producing the `N/2 + 1` amplitudes from DC to Nyquist.
    - Can pre-process the signals with a windowing function (e.g. Hann, Blackman), applied
      in periodic (sym=False) form as in `scipy.signal.periodogram`.
    - Scales the amplitudes under scipy's names: `amplitude` reads a tone's peak amplitude A,
      `amplitude_squared` reads A² (the square of `amplitude`, a convenience), `spectrum` its
      mean-square A²/2 (`scipy.signal.periodogram(scaling="spectrum")`), and `density` its
      power density (`scaling="density"`, integrates to the signal's mean-square power).
    - The length of each signal must be a power of two of at least 2 — a violation raises —
      and the corresponding frequency axis is
      `[i * sample_rate / N for i in range(N // 2 + 1)]`.

- `polist.agg_slices`
    - Computes an aggregation of a range of y-values defined by some x-values for List-type columns.
    - This is useful for feature extraction from signals, e.g. to compute the mean of a signal in a
      certain time range or a spectrum in a certain frequency range.
    - Ranges to include or exclude can be given as simple `(lo, hi)` tuples or with explicit
      boundary modes like `((lo, "closed"), (hi, "open"))`.

- `polist.agg_lists`
    - Applies element-wise list-aggregations to a List-type column in a GroupBy context.
    - Composed from Polars' own vertical aggregations, with a small pass-through plugin
      that normalizes `Array` to `List` and carries the `strict`/`list_length` checks; it
      lives here so that both aggregation functions offer the same aggregations with the
      same missing-data behaviour. One consequence: it aggregates in the column's own
      dtype, as Polars does, rather than in `f64` -- so near the `Float32` range limit it
      can overflow where the round-once plugin kernels stay finite.

- `polist.zip_binary`
    - Applies a binary operation element-wise between two List-type columns, per element
      exactly as the corresponding scalar Polars operation: arithmetic (`add`, `sub`, `mul`,
      `div`) with null propagation and IEEE float behaviour (output dtype follows Polars
      supertyping, so two `Float32` inputs stay `Float32`), `and`/`or` on Boolean lists with
      Kleene logic, and comparisons (`gt`, `ge`, `lt`, `le`, `eq`, `ne`) following Polars'
      total order (NaN equals NaN and exceeds everything else), emitting Boolean lists.

- `polist.cum_agg_runs`
    - Cumulatively aggregates a List-type column within runs gated by a Boolean List-type
      column: a run is a maximal region of constant gate value (null gates break runs), only
      `True`-gated elements accumulate, and every other position emits null (or a plain zero
      with `outside="zero"`).
    - Each emitted element equals the vertical aggregation over the run's elements so far,
      so an all-`True` gate over `Float64` tracks `cum_sum`/`cum_min`/`cum_max`/`cum_count`;
      null values emit null but keep the running state, except `count`, which emits the
      running count as `cum_count` does.
    - This function has always emitted the true prefix extreme -- NaN for an all-NaN
      prefix, or a leading infinity -- where Polars up to 1.43 leaked its `cum_min`/
      `cum_max` fold seed (`+f64::MAX` for `cum_min`, `-f64::MAX` for `cum_max`; briefly
      `±inf`) at those positions. Polars >= 1.44 agrees with this function. `Float32`
      values accumulate in `f64` and round once at the boundary, so a Polars version that
      accumulates step-wise in `Float32` can differ in the last bits and near the range
      limit.
    - `Float32` values stay `Float32`; `count` emits `UInt32` (Polars' count dtype).

Every function accepts a length-1 literal list column (e.g. `pl.lit([...])`) for any
input and broadcasts it across rows, provided at least one other input is a real column --
as in Polars generally, an expression built only from length-1 literals produces a single
row rather than broadcasting, so the claim is moot for the single-input transforms and for
`agg_lists`, which aggregates rows rather than mapping them. Anything structural raises instead of silently
returning nulls: invalid configuration (bad cutoffs, unknown window names), mismatched
paired lengths, integer inner dtypes, and per-row shapes a transform cannot process — a
non-power-of-two FFT length, a Butterworth signal shorter than its reflection padding, an
empty list. For mixed-length List data the escape hatch is pre-filtering with `list.len()`
*before* the transform — in a lazy query the optimizer pushes a later filter below the
transform, so offending rows are dropped before validation rather than raising.
Only genuinely missing data nulls: a null row, or a null element inside a transform's
signal.

`Float32` columns stay `Float32`; the plugin kernels compute in `f64` and round once at
the output boundary, while `agg_lists` aggregates in the column's own dtype as Polars
does. The one dtype exception: `count` emits `UInt32`, Polars' count dtype, everywhere it
appears.

Both aggregation functions support `sum`, `mean`, `median`, `std`, `min`, `max`, `delta`,
and `count`, and both handle missing data like the vertical aggregations of Polars itself.
(Semantically, that is: floating-point accumulation order differs from Polars', so results
agree to rounding rather than bit-for-bit.)

- Nulls are skipped: excluded from both the numerator and the denominator, so `count`
  only counts non-null values and an empty or all-null selection yields null, `count`
  reading 0 and `sum` following the `empty` setting below.
- NaNs propagate: Polars treats NaN as a legitimate float value, so it flows into the
  sum and poisons `sum`, `mean` and `std`. Two exceptions to be aware of are `min` and `max`,
  which skip NaNs (unless all values are NaN), and `median` sorts NaN as the largest value.
- `std` is the sample standard deviation (ddof=1) and yields null for fewer than two
  values, as in Polars.

The three transforms instead follow a three-line rule: missing data nulls (a null row or
a null element yields a null row — a signal with missing samples cannot be meaningfully
transformed); invalid values propagate (NaN and ±inf flow through the arithmetic — one NaN
yields an all-NaN spectrum, staying visibly invalid rather than becoming missing); wrong
shape raises. For the aggregations an empty list is simply an empty selection, and
`zip_binary` and `cum_agg_runs` mirror the corresponding scalar and cumulative Polars
operations per element, treating empty lists as valid (empty in, empty out).

### List and Array

Every function accepts both `List` and `Array` columns. A length-preserving function
returns the container it was given; `apply_fft` narrows an `Array(w)` to `Array(w // 2 + 1)`,
and `apply_interp` follows its `xp` argument, since that decides the output's shape.
`zip_binary` returns an `Array` only when both operands are `Array`s, since both are
operands; `cum_agg_runs` follows its value column alone, its gate being a mask rather than
an operand. Where two inputs must agree element-for-element, their widths are checked
before execution if both are `Array`s, and per row otherwise. Two exceptions to container
propagation: `agg_slices` reduces to a scalar, and `agg_lists` always returns a `List`,
being an expression composition that never sees its input's schema (an `Array` input to it
must match `list_length` exactly, rather than being null-padded like a short `List`).

Zero-width `Array`s are rejected at any nesting depth, because Polars panics when slicing
them; zero-length `List`s are fine.

### Two conventions, one switch

Polars answers "what is the sum of no values" with `0.0`, the identity element of addition.
For feature extraction that is the dangerous answer: an empty selection usually means a
misconfigured range, and a real `0.0` disguises that as a measurement which then sails
through a threshold. So `agg_slices` and `agg_lists` take an `empty` argument:

- `empty="null"` (default) — summing nothing is unknown, not zero.
- `empty="zero"` — Polars' convention, matching
  `explode().group_by().agg(col.filter(...).sum())`.

It reaches exactly one cell of the matrix: `sum` is the only aggregation with a non-null
identity, every other one yields null over nothing either way, and `count` yields 0 either
way.

The switch governs an empty *selection* — no index fell in range, or the lists were
themselves empty, the limiting case of the same condition. A null row stays null under both
settings for every aggregation, `count` included. So `empty="zero"` is not the same as
`.fill_null(0.0)`: that cannot tell a missing row from an empty selection, and would turn a
missing measurement into a real zero, which is the error this flag exists to avoid.

The same philosophy sets the `strict` default. `apply_interp` verifies that `x` is
non-decreasing, `agg_slices` rejects inverted or NaN range bounds at expression
construction, and `agg_lists` raises on lists carrying non-null data past
`list_length` rather than silently truncating it (a null-padded tail is lossless and
never raises) — all **on by default**, since each covers a failure that is
otherwise silent. Pass `strict=False` for the lenient readings (numpy's unsorted-`x`
behaviour, empty selections from inverted ranges, `list_length` as a deliberate window).
NaN, which announces itself in the output, propagates rather than raising either way.

Errors raised from inside a row describe the offending row by its shape rather than its
position: Polars hands a plugin one chunk at a time, so any row number counted there would
be chunk-local and would point at a different, valid row of the frame.

### Example

```python
import polars as pl
import polars_list_utils as polist

Fs, N = 200.0, 1024
FREQS = [i * Fs / N for i in range(N // 2 + 1)]

df = (
    df
    .with_columns(
        polist.apply_butterworth(
            "signal",
            sample_rate=Fs,
            max_freq=50.0,
        )
        .alias("filtered")
    )
    .with_columns(
        polist.apply_fft(
            "filtered",
            sample_rate=Fs,
            window="hann",
            scaling="amplitude",
        )
        .alias("fft")
    )
    .with_columns(
        polist.agg_slices(
            "fft",
            pl.lit(FREQS),
            aggregation="mean",
            slices_include=[(20.0, 30.0)],
        )
        .alias("mean_20_30hz")
    )
)
```

See [examples/showcase.py](https://github.com/dashdeckers/polars_list_utils/blob/main/examples/showcase.py) for the full pipeline:

![Showcase](https://raw.githubusercontent.com/dashdeckers/polars_list_utils/main/examples/showcase.png)

## Installation (user)

```bash
uv pip install polars-list-utils
```

## Installation (developer)

1) Setup Rust (i.e. install rustup)
2) Setup Python (i.e. install uv)
3) Setup environment and compile plugin:

```bash
uv sync --extra dev
uv run maturin develop --release --uv
```

4) (Maybe) configure Cargo to find uv's Python installs. For example:

```
# .cargo/config.toml
[env]
PYO3_PYTHON = "C:\\Users\\travis.hammond\\AppData\\Roaming\\uv\\python\\cpython-3.12.0-windows-x86_64-none\\python.exe"
```

5) Test and run the example:

```bash
uv run pytest
uv run ./examples/showcase.py
```

6) Lint:

```bash
uvx ty check
uvx ruff check
cargo clippy --release -- -D warnings
cargo fmt
```

## Changelog

### 2.0.0

One bundled breaking release. New: the `sum` aggregation everywhere; `zip_binary`
(element-wise arithmetic, Kleene `and`/`or`, and total-ordered comparisons between two
list columns); `cum_agg_runs` (cumulative aggregation within gate-delimited runs, matching
Polars' `cum_*` semantics); `Array` columns accepted by every function, with the container
preserved; and an `empty` switch on the aggregations choosing what the sum of nothing is
-- `"null"` (default) or Polars' `"zero"`.

Breaking: `strict` checks are on by default (unsorted interpolation axes, invalid ranges,
and lossy `agg_lists` truncation now raise; pass `strict=False` for the old leniency).
Wrong shape raises instead of yielding null rows (non-power-of-two FFT lengths, too-short
Butterworth signals, empty lists into transforms -- pre-filter with `list.len()` upstream),
while NaN/±inf now propagate into visibly invalid output instead of becoming null: guard
thresholds with `is_finite()`. `Float32` columns stay `Float32`, `count` returns `UInt32`,
and integer inner dtypes raise. FFT scalings use scipy's names -- `power` -> `spectrum`,
`psd` -> `density`, plus new `amplitude_squared`; window names are unchanged. The
even-count median now matches Polars exactly (last-ulp changes on stored features).

Fixed: `Array` inputs no longer abort the process; no input dtype can panic; aliased or
JSON-shaped range arguments no longer crash `agg_slices`.
