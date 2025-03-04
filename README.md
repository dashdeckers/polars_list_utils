# Polars List Utils (`polist`)

`polist` is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames.

Status: Work-in-Progress!

## Features

- `dsp` - Basic digital signal processing including Fast Fourier Transform (FFT), windowing, and Butterworth filtering
- `agg` - Elementwise aggregations for List-type columns (currently sum, mean and count)
- `feat` - Feature extraction for List-type columns, currently only mean_of_range

## Installation (user)

```bash
uv pip install polars-list-utils
```

## Installation (developer)

1) Setup your Python environment according to the pyproject.toml file
2) Setup your Rust environment
3) Compile:

```bash
uv sync --extra dev
uv run maturin develop --release
```

4) Run:

```bash
uv venv
.venv\Scripts\activate
python .\scripts\showcase_fft.py
```

5) Maybe configure Cargo to find uv Pythons. For example:

```
# .cargo/config.toml
[env]
PYO3_PYTHON = "C:\\Users\\travis.hammond\\AppData\\Roaming\\uv\\python\\cpython-3.12.0-windows-x86_64-none\\python.exe"
```

## Todo

- Add more features
- Add more tests