from typing import Union
from pathlib import Path

import polars as pl
import polars.selectors as cs
import polars_list_utils as polist
import numpy as np
import matplotlib.pyplot as plt

from polars_list_utils._internal import __version__ as __version__
print(__version__)

Fs = 200  # Hertz
t = 6  # Seconds

def generate_sine_wave(
    freq: list[Union[int, float]],
    sample_rate: Union[int, float],
    duration: Union[int, float],
):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    y = 0
    for f in freq:
        y += np.sin((2 * np.pi) * (x * f))
    return y


df = pl.DataFrame({
    'signal': [
        [e for e in generate_sine_wave([1], Fs, t)[:1024]],
        [e for e in generate_sine_wave([80], Fs, t)[:1024]],
        [e for e in generate_sine_wave([65], Fs, t)[:1024]],
        [e for e in generate_sine_wave([40], Fs, t)[:1024]],
        [e for e in generate_sine_wave([10], Fs, t)[:1024]],
        [e for e in generate_sine_wave([80, 60, 40, 10], Fs, t)[:1024]],
    ],
    "norm_col": [1.0, 80.0, 65.0, 40.0, 10.0, 10.0]
})
print(df)

# shape: (6, 2)
# ┌──────────────────────────────┬──────────┐
# │ signal                       ┆ norm_col │
# │ ---                          ┆ ---      │
# │ list[f64]                    ┆ f64      │
# ╞══════════════════════════════╪══════════╡
# │ [0.0, 0.031411, … 0.661312]  ┆ 1.0      │
# │ [0.0, 0.587785, … 0.951057]  ┆ 80.0     │
# │ [0.0, 0.891007, … 0.156434]  ┆ 65.0     │
# │ [0.0, 0.951057, … -0.587785] ┆ 40.0     │
# │ [0.0, 0.309017, … 0.809017]  ┆ 10.0     │
# │ [0.0, 2.798915, … 0.584503]  ┆ 10.0     │
# └──────────────────────────────┴──────────┘

df_plot = (
    df
    .with_columns(
        polist.apply_fft(
            list_column='signal',
            sample_rate=Fs,
            window="hanning",
            skip_fft=True,
        ).alias('signal'),
    )
    .with_columns(
        polist.apply_fft(
            list_column='signal',
            sample_rate=Fs,
        ).alias('fft'),
    )
    .with_columns(
        polist.get_freqs(
            list_column='signal',
            sample_rate=Fs,
        ).alias('freqs'),
    )
    .with_columns(
        polist.normalize_fft(
            list_column='fft',
            norm_column='norm_col',
            max_norm_val=10,
            sample_rate=Fs,
        ).alias('nrm_fft'),
    )
    .with_columns(
        polist.get_normalized_freqs(
            list_column='fft',
            max_norm_val=10,
        ).alias('nrm_freqs'),
    )
)
print(df_plot)

# shape: (6, 6)
# ┌─────────────┬──────────┬─────────────┬─────────────┬────────────┬────────────┐
# │ signal      ┆ norm_col ┆ fft         ┆ freqs       ┆ nrm_fft    ┆ nrm_freqs  │
# │ ---         ┆ ---      ┆ ---         ┆ ---         ┆ ---        ┆ ---        │
# │ list[f64]   ┆ f64      ┆ list[f64]   ┆ list[f64]   ┆ list[f64]  ┆ list[f64]  │
# ╞═════════════╪══════════╪═════════════╪═════════════╪════════════╪════════════╡
# │ [0.0,       ┆ 1.0      ┆ [0.171077,  ┆ [0.0,       ┆ [0.171077, ┆ [0.0,      │
# │ 2.9565e-7,  ┆          ┆ 0.369423, … ┆ 0.1953125,  ┆ 0.190931,  ┆ 0.019531,  │
# │ … 0.000006] ┆          ┆ 1.0021e…    ┆ … 100.0]    ┆ … 0.00044… ┆ … 10.0]    │
# │ [0.0,       ┆ 80.0     ┆ [0.000002,  ┆ [0.0,       ┆ [0.000002, ┆ [0.0,      │
# │ 0.000006, … ┆          ┆ 0.000002, … ┆ 0.1953125,  ┆ 0.000002,  ┆ 0.019531,  │
# │ 0.000009]   ┆          ┆ 0.00013…    ┆ … 100.0]    ┆ … 0.00013… ┆ … 10.0]    │
# │ [0.0,       ┆ 65.0     ┆ [0.000001,  ┆ [0.0,       ┆ [0.000001, ┆ [0.0,      │
# │ 0.000008, … ┆          ┆ 0.000001, … ┆ 0.1953125,  ┆ 0.000001,  ┆ 0.019531,  │
# │ 0.000001]   ┆          ┆ 0.00001…    ┆ … 100.0]    ┆ … 0.00001… ┆ … 10.0]    │
# │ [0.0,       ┆ 40.0     ┆ [0.000006,  ┆ [0.0,       ┆ [0.000006, ┆ [0.0,      │
# │ 0.000009, … ┆          ┆ 0.000006, … ┆ 0.1953125,  ┆ 0.000007,  ┆ 0.019531,  │
# │ -0.000006]  ┆          ┆ 0.00000…    ┆ … 100.0]    ┆ … 0.00000… ┆ … 10.0]    │
# │ [0.0,       ┆ 10.0     ┆ [0.00042,   ┆ [0.0,       ┆ [0.00042,  ┆ [0.0,      │
# │ 0.000003, … ┆          ┆ 0.000422, … ┆ 0.1953125,  ┆ 0.000422,  ┆ 0.019531,  │
# │ 0.000008]   ┆          ┆ 2.6399e-…   ┆ … 100.0]    ┆ …          ┆ … 10.0]    │
# │             ┆          ┆             ┆             ┆ 2.6399e-…  ┆            │
# │ [0.0,       ┆ 10.0     ┆ [0.000429,  ┆ [0.0,       ┆ [0.000429, ┆ [0.0,      │
# │ 0.000026, … ┆          ┆ 0.000432, … ┆ 0.1953125,  ┆ 0.000432,  ┆ 0.019531,  │
# │ 0.000006]   ┆          ┆ 0.00014…    ┆ … 100.0]    ┆ … 0.00014… ┆ … 10.0]    │
# └─────────────┴──────────┴─────────────┴─────────────┴────────────┴────────────┘

print(df_plot.with_columns(
    pl.col("signal").list.len().alias("signal_len"),
    pl.col("fft").list.len().alias("fft_len"),
    pl.col("freqs").list.len().alias("freqs_len"),
    pl.col("nrm_fft").list.len().alias("nrm_fft_len"),
    pl.col("nrm_freqs").list.len().alias("nrm_freqs_len"),
).select(cs.ends_with("len")))

# shape: (6, 5)
# ┌────────────┬─────────┬───────────┬─────────────┬───────────────┐
# │ signal_len ┆ fft_len ┆ freqs_len ┆ nrm_fft_len ┆ nrm_freqs_len │
# │ ---        ┆ ---     ┆ ---       ┆ ---         ┆ ---           │
# │ u32        ┆ u32     ┆ u32       ┆ u32         ┆ u32           │
# ╞════════════╪═════════╪═══════════╪═════════════╪═══════════════╡
# │ 1024       ┆ 513     ┆ 513       ┆ 513         ┆ 513           │
# │ 1024       ┆ 513     ┆ 513       ┆ 513         ┆ 513           │
# │ 1024       ┆ 513     ┆ 513       ┆ 513         ┆ 513           │
# │ 1024       ┆ 513     ┆ 513       ┆ 513         ┆ 513           │
# │ 1024       ┆ 513     ┆ 513       ┆ 513         ┆ 513           │
# │ 1024       ┆ 513     ┆ 513       ┆ 513         ┆ 513           │
# └────────────┴─────────┴───────────┴─────────────┴───────────────┘

fig, axs = plt.subplots(
    nrows=3,
    ncols=len(df_plot),
    squeeze=False,
    figsize=(5 * len(df_plot), 12),
)
for i in range(len(df_plot)):
    axs[0][i].plot(
        df_plot[i, 'signal'].to_numpy(),
    )
    axs[1][i].plot(
        df_plot[i, 'freqs'].to_numpy(),
        df_plot[i, 'fft'].to_numpy(),
    )
    axs[2][i].plot(
        df_plot[i, 'nrm_freqs'].to_numpy(),
        df_plot[i, 'nrm_fft'].to_numpy(),
    )
plt.tight_layout()
plt.savefig(Path.cwd() / "examples" / "showcase_dsp.png")
# plt.show()

