# Polars List Utils (`polist`)

`polist` is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames.

So far these utilities comprise those that I found to be missing or lacking from
the List namespace within the Polars library while I was working on a project at
work that required extensive handling signal data which I was storing in Polars
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

- `dsp` - Basic digital signal processing including Fast Fourier Transform (FFT), windowing, and Butterworth filtering
- `agg` - Elementwise aggregations for List-type columns (currently sum, mean and count)
- `feat` - Feature extraction for List-type columns (currently only mean_of_range)

### Example: (signal) -- (hann window) -- (FFT) -- (Freq. Normalization)

![DSP Example](examples/showcase_dsp.png)


## Installation (user)

```bash
uv pip install polars-list-utils
```

## Installation (developer)

1) Setup Python (i.e. install uv)
2) Setup Rust (i.e. install rustup)
3) Compile:

```bash
uv venv
uv sync --extra dev
uv run maturin develop --release
```

4) Run:

```bash
.venv\Scripts\activate
python .\scripts\showcase_dsp.py
deactivate
```

5) Maybe configure Cargo to find uv Pythons. For example:

```
# .cargo/config.toml
[env]
PYO3_PYTHON = "C:\\Users\\travis.hammond\\AppData\\Roaming\\uv\\python\\cpython-3.12.0-windows-x86_64-none\\python.exe"
```

6) Lint

```bash
uvx ruff check
cargo fmt
```

## Todo

- Add more features
- Add more tests