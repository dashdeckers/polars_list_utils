# Polars List Utils (`polist`)

`polist` is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames.

## Features

- `dsp` - Basic digital signal processing including Fast Fourier Transform (FFT), windowing, and Butterworth filtering
- `agg` - Elementwise aggregations for List-type columns (currently sum, mean and count)
- `feat` - Feature extraction for List-type columns, currently only mean_of_range

## Installation

1) Setup your Python environment according to the pyproject.toml file
2) Setup your Rust environment
3) Compile:

```bash
uv sync
uv run maturin develop --release
```

4) Run:

```bash
uv venv
.venv\Scripts\activate
python .\scripts\showcase.py
```

## Todo

- Publish to PyPI (uv?)
- Add more features
- Add more tests