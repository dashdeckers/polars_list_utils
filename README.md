# Polars List Utils (`polist`)

`polist` is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames, especially for signal processing and feature extraction.

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

Status: Work-in-Progress!

## Features

- `polist.apply_fft`
    - Applies a Fast Fourier Transform (FFT) to a List-type column of signal data.
    - Can pre-process the signals with a windowing function (e.g. Hann, Hamming, Blackman).
    - Can pre-process the signals with a Butterworth filter (low-pass, high-pass, band-pass).
    - Can normalize the FFT amplitudes by the signal length or the window sum.

- `polist.operate_scalar_on_list`
    - Applies a scalar operation of a Float-type column to each element of a List-type column.
    - This is currently not supported in Polars, see [this issue][list_eval_named].
    - Can apply the operations `add`, `sub`, `mul`, and `div`.

- `polist.interpolate_columns`
    - Interpolates a new List-type column from 3 specified List-type columns.
    - Behaviour as expected from the `numpy.interp` function, but for Polars DataFrames.
    - Supply the `x_data`, `y_data`, and `x_interp` columns to obtain the `y_interp` column.

- `polist.aggregate_list_col_elementwise`
    - Applies element-wise list-aggregations to a List-type column in a GroupBy context.
    - Currently supports `sum`, `mean`, and `count`.
    - This is possible using the Polars API e.g. using `list.get(n)` (see my SO question [here][elementwise_agg]), but it does not scale well as for large lists and complicated queries it can lead to a stack overflow (see [this issue][stack_overflow] and many others).

- `polist.mean_of_range`
    - Computes the mean of a range of y-values defined by some x-values for List-type columns.
    - This is useful for feature extraction from signals, e.g. to compute the mean of a signal in a certain time range or a spectrum in a certain frequency range.
    - This is somewhat possible using the Polars API (e.g. using `list.slice` and `list.mean`), but can get very complicated for the simple case of wanting to specify certain y-values based on a custom x-axis.

- `polist.fft_freqs`
    - Computes the frequencies of the FFT bins for a given sampling rate and number of samples.
    - This function does not operate dynamically on List-type columns, use it with `pl.lit`.
    - This is useful for plotting the FFT spectrum in the frequency domain. Similar to the `numpy.fft.fftfreq` function, but in my opinion much simpler.

- `polist.fft_freqs_linspace`
    - Basically a thin wrapper around something like `numpy.linspace` to create a linearly spaced List of values.
    - This function does not operate dynamically on List-type columns, use it with `pl.lit`.
    - This is useful when used together with `polist.interpolate_columns`.

[list_eval_named]: https://github.com/pola-rs/polars/issues/7210
[elementwise_agg]: https://stackoverflow.com/questions/73776179/element-wise-aggregation-of-a-column-of-type-listf64-in-polars
[stack_overflow]: https://github.com/pola-rs/polars/issues/5455


### Example: (signal) -- (hann window) -- (FFT) -- (Freq. Normalization)

![DSP Example](examples/showcase_dsp.png)


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
uv run maturin develop --release
```

4) (Maybe) configure Cargo to find uv's Python installs. For example:

```
# .cargo/config.toml
[env]
PYO3_PYTHON = "C:\\Users\\travis.hammond\\AppData\\Roaming\\uv\\python\\cpython-3.12.0-windows-x86_64-none\\python.exe"
```

5) Run:

```bash
uv run ./examples/showcase_dsp.py
```

6) Lint

```bash
uvx ruff check
cargo fmt
```

## Todo

- Add more features
- Add more tests