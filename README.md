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
    - Can pre-process the signals with a windowing function (e.g. Hann, Blackman).
    - Can scale the amplitudes following the scipy conventions: `amplitude` reads a tone's peak
      amplitude, `power` its mean-square (`scipy.signal.periodogram(scaling="spectrum")`), and
      `psd` its power density (`scaling="density"`, integrates to the signal's mean-square power).
    - The length of each signal must be a power of two, and the corresponding frequency axis
      is given by `[i * sample_rate / N for i in range(N // 2 + 1)]`.

- `polist.agg_slices`
    - Computes an aggregation of a range of y-values defined by some x-values for List-type columns.
    - This is useful for feature extraction from signals, e.g. to compute the mean of a signal in a
      certain time range or a spectrum in a certain frequency range.
    - Ranges to include or exclude can be given as simple `(lo, hi)` tuples or with explicit
      boundary modes like `((lo, "closed"), (hi, "open"))`.

- `polist.agg_lists`
    - Applies element-wise list-aggregations to a List-type column in a GroupBy context.
    - This is implemented in pure Polars (no plugin), and lives here so that both aggregation
      functions offer the same aggregations with the same missing-data behaviour.

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
    - Two documented departures from the native cumulatives: `cum_min`/`cum_max` fold from
      ∓`f64::MAX` and never replace that seed on a tie, so they report ±1.8e308 where the
      prefix is all-NaN or opens with an infinity of the same sign, while this function
      reports the true prefix extreme; and `Float32` values accumulate in `f64` and round
      once at the boundary, where Polars rounds every step.
    - `Float32` values stay `Float32`; `count` emits `UInt32` (Polars' count dtype).

The six plugin functions accept a length-1 literal list column (e.g. `pl.lit([...])`) for
any input and broadcast it across rows. Invalid configuration (bad cutoffs, unknown window
names, mismatched list lengths) raises an error instead of silently returning nulls, while
per-row data problems (e.g. a signal length that is not a power of two) yield null rows.

Both aggregation functions support `sum`, `mean`, `median`, `std`, `min`, `max`, `delta`,
and `count`, and both handle missing data like the vertical aggregations of Polars itself.
(Semantically, that is: floating-point accumulation order differs from Polars', so results
agree to rounding rather than bit-for-bit.)

- Nulls are skipped: excluded from both the numerator and the denominator, so `count`
  only counts non-null values and an empty or all-null selection yields null (`count`: 0,
  `sum`: 0.0 — Polars' identity for sum).
- NaNs propagate: Polars treats NaN as a legitimate float value, so it flows into the
  sum and poisons `sum`, `mean` and `std`. Two exceptions to be aware of are `min` and `max`,
  which skip NaNs (unless all values are NaN), and `median` sorts NaN as the largest value.
- `std` is the sample standard deviation (ddof=1) and yields null for fewer than two
  values, as in Polars.

The three transforms instead turn any row with missing data (a null row, an empty list, or
a list containing nulls) into a null output row: a signal with missing samples cannot be
meaningfully transformed. `zip_binary` and `cum_agg_runs` mirror the corresponding scalar
and cumulative Polars operations per element instead, and treat empty lists as valid
(empty in, empty out).

Some checks are opt-in via `strict=True` (default off): `apply_interp` verifies that `x`
is non-decreasing, `agg_slices` rejects inverted or NaN range bounds at expression
construction, and `agg_lists` raises on lists longer than `list_length` instead of
silently truncating them. Each check covers a failure that is otherwise silent; NaN, which
announces itself in the output, propagates rather than raising.

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
