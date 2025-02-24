# Polars List Utils (`polist`)

`polist` is a Python package that provides a set of utilities for working with List-type columns in Polars DataFrames.

## Features

- `fft` - Fast Fourier Transform

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
    - [ ] Aggregations
    - [ ] Piecewise Mean & Relation
- Add more tests