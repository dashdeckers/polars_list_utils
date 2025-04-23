from typing import Union
from pathlib import Path

import polars as pl
import polars.selectors as cs
import polars_list_utils as polist
import numpy as np
import matplotlib.pyplot as plt

from polars_list_utils._internal import __version__ as __version__
print(__version__)

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


Fs = 200        # Sample-rate [Hz]
t = 6           # Sample-duration [s]

N = 1024        # Signals bins at nearest power of two [-]
n = (N / 2) + 1 # Spectrum bins resulting from (real-valued) FFT [-]

fmax_i = 10     # Maximum frequency for interpolation [Hz]


df_plot = (
    # Create a DataFrame with some dummy signals (generated sine waves)
    pl.DataFrame({
        'S': [
            [e for e in generate_sine_wave([1], Fs, t)[:N]],
            [e for e in generate_sine_wave([10], Fs, t)[:N]],
            [e for e in generate_sine_wave([40], Fs, t)[:N]],
            [e for e in generate_sine_wave([65], Fs, t)[:N]],
            [e for e in generate_sine_wave([80], Fs, t)[:N]],
            [e for e in generate_sine_wave([80, 60, 40, 10], Fs, t)[:N]],
        ],
        # Note some frequencies to normalize by later
        'f': [1.0, 10.0, 40.0, 65.0, 80.0, 10.0],
    })
    # Transform the signal and skip FFT (to plot it later)
    .with_columns(
        polist.apply_fft(
            list_column='S',        # Signal
            sample_rate=Fs,
            window="hanning",
            bp_min=0,
            bp_max=Fs / 2,
            bp_ord=4,
            skip_fft=True,
        ).alias('S_t'),             # Signal transformed
    )
    # Transform the signal and compute the FFT
    .with_columns(
        polist.apply_fft(
            list_column='S',        # Signal
            sample_rate=Fs,
            window="hanning",
            bp_min=0,
            bp_max=Fs / 2,
            bp_ord=4,
            norm="window",
        ).alias('A'),               # FFT Amplitude
    )
    # Compute the corresponding frequencies for the FFT amplitudes
    .with_columns(
        pl.lit(polist.fft_freqs(
            n=int(n),
            fs=Fs,
        )).alias('F')               # FFT Frequency
    )
    # Normalize the FFT frequencies by their average frequency
    .with_columns(
        polist.operate_scalar_on_list(
            list_column='F',        # FFT Frequency
            scalar_column='f',      # Frequency to normalize by
            operation='div',
        ).alias('F_n'),             # Normalized FFT Frequency
    )
    # Compute the average amplitude of frequencies lower than the
    # first harmonic, but excluding the DC component
    .with_columns(
        polist.mean_of_range(
            list_column_y='A',      # FFT Amplitude
            list_column_x='F_n',    # Normalized FFT Frequency
            # Take y_values where x_axis is in [0, 1)
            x_min=0,
            x_max=1,
            # Skip the first index (DC component)
            x_min_idx_offset=1,
        ).alias('Avg(A_n[0:1])'),   # Average amplitude before the first harmonic
    )                               # (DC component excluded)
    # Interpolate the FFT amplitudes to a common x-axis of normalized
    # frequencies (e.g. a linear space from 0 to 10) for easier plotting
    # and aggregating
    .with_columns(
        pl.lit(polist.fft_freqs_linspace(
            fnum=int(n),
            fmax=fmax_i,
        )).alias('F_i')             # FFT Frequency to interpolate to
    )
    .with_columns(
        polist.interpolate_columns(
            x_data='F_n',           # Normalized FFT Frequency
            y_data='A',             # FFT Amplitude
            x_interp='F_i',         # FFT Frequency to interpolate to
        ).alias('A_i')              # Interpolated FFT Amplitude
    )
)
with pl.Config(tbl_cols=-1):
    print(df_plot)


# shape: (6, 9)
# ┌───────────────────┬──────┬────────────────────┬───────────────────────────┬────────────────────┬───────────────────────────┬───────────────┬──────────────────────────┬──────────────────────────┐
# │ S                 ┆ f    ┆ S_t                ┆ A                         ┆ F                  ┆ F_n                       ┆ Avg(A_n[0:1]) ┆ F_i                      ┆ A_i                      │
# │ ---               ┆ ---  ┆ ---                ┆ ---                       ┆ ---                ┆ ---                       ┆ ---           ┆ ---                      ┆ ---                      │
# │ list[f64]         ┆ f64  ┆ list[f64]          ┆ list[f64]                 ┆ list[f64]          ┆ list[f64]                 ┆ f64           ┆ list[f64]                ┆ list[f64]                │
# ╞═══════════════════╪══════╪════════════════════╪═══════════════════════════╪════════════════════╪═══════════════════════════╪═══════════════╪══════════════════════════╪══════════════════════════╡
# │ [0.0, 0.031411, … ┆ 1.0  ┆ [0.0, 2.9565e-7, … ┆ [0.000667, 0.00144, …     ┆ [0.0, 0.1953125, … ┆ [0.0, 0.1953125, … 100.0] ┆ 0.284072      ┆ [0.0, 0.019493, … 10.0]  ┆ [0.000667, 0.000744, …   │
# │ 0.661312]         ┆      ┆ 0.000006]          ┆ 3.9069e-…                 ┆ 100.0]             ┆                           ┆               ┆                          ┆ 0.00000…                 │
# │ [0.0, 0.309017, … ┆ 10.0 ┆ [0.0, 0.000003, …  ┆ [0.000002, 0.000002, …    ┆ [0.0, 0.1953125, … ┆ [0.0, 0.019531, … 10.0]   ┆ 0.026697      ┆ [0.0, 0.019493, … 10.0]  ┆ [0.000002, 0.000002, …   │
# │ 0.809017]         ┆      ┆ 0.000008]          ┆ 1.0292e…                  ┆ 100.0]             ┆                           ┆               ┆                          ┆ 1.0292e…                 │
# │ [0.0, 0.951057, … ┆ 40.0 ┆ [0.0, 0.000009, …  ┆ [2.5254e-8, 2.5263e-8, …  ┆ [0.0, 0.1953125, … ┆ [0.0, 0.004883, … 2.5]    ┆ 0.003496      ┆ [0.0, 0.019493, … 10.0]  ┆ [2.5254e-8, 2.5397e-8, … │
# │ -0.587785]        ┆      ┆ -0.000006]         ┆ 7.036…                    ┆ 100.0]             ┆                           ┆               ┆                          ┆ 7.036…                   │
# │ [0.0, 0.891007, … ┆ 65.0 ┆ [0.0, 0.000008, …  ┆ [5.3433e-9, 5.3442e-9, …  ┆ [0.0, 0.1953125, … ┆ [0.0, 0.003005, …         ┆ 0.002148      ┆ [0.0, 0.019493, … 10.0]  ┆ [5.3433e-9, 5.3809e-9, … │
# │ 0.156434]         ┆      ┆ 0.000001]          ┆ 3.789…                    ┆ 100.0]             ┆ 1.538462]                 ┆               ┆                          ┆ 3.789…                   │
# │ [0.0, 0.587785, … ┆ 80.0 ┆ [0.0, 0.000006, …  ┆ [5.9616e-9, 5.9619e-9, …  ┆ [0.0, 0.1953125, … ┆ [0.0, 0.002441, … 1.25]   ┆ 0.002309      ┆ [0.0, 0.019493, … 10.0]  ┆ [5.9616e-9, 5.9813e-9, … │
# │ 0.951057]         ┆      ┆ 0.000009]          ┆ 5.349…                    ┆ 100.0]             ┆                           ┆               ┆                          ┆ 5.349…                   │
# │ [0.0, 2.798915, … ┆ 10.0 ┆ [0.0, 0.000026, …  ┆ [0.000002, 0.000002, …    ┆ [0.0, 0.1953125, … ┆ [0.0, 0.019531, … 10.0]   ┆ 0.026697      ┆ [0.0, 0.019493, … 10.0]  ┆ [0.000002, 0.000002, …   │
# │ 0.584503]         ┆      ┆ 0.000006]          ┆ 5.6825e…                  ┆ 100.0]             ┆                           ┆               ┆                          ┆ 5.6825e…                 │
# └───────────────────┴──────┴────────────────────┴───────────────────────────┴────────────────────┴───────────────────────────┴───────────────┴──────────────────────────┴──────────────────────────┘

print(df_plot.with_columns(
    pl.col("F").list.len().alias("F_len"),
    pl.col("A").list.len().alias("A_len"),
    pl.col("F_i").list.len().alias("F_i_len"),
    pl.col("A_i").list.len().alias("A_i_len"),
    pl.col("S").list.len().alias("S_len"),
).select(cs.ends_with("_len")))

# ┌───────┬───────┬─────────┬─────────┬───────┐
# │ F_len ┆ A_len ┆ F_i_len ┆ A_i_len ┆ S_len │
# │ ---   ┆ ---   ┆ ---     ┆ ---     ┆ ---   │
# │ u32   ┆ u32   ┆ u32     ┆ u32     ┆ u32   │
# ╞═══════╪═══════╪═════════╪═════════╪═══════╡
# │ 513   ┆ 513   ┆ 513     ┆ 513     ┆ 1024  │
# │ 513   ┆ 513   ┆ 513     ┆ 513     ┆ 1024  │
# │ 513   ┆ 513   ┆ 513     ┆ 513     ┆ 1024  │
# │ 513   ┆ 513   ┆ 513     ┆ 513     ┆ 1024  │
# │ 513   ┆ 513   ┆ 513     ┆ 513     ┆ 1024  │
# │ 513   ┆ 513   ┆ 513     ┆ 513     ┆ 1024  │
# └───────┴───────┴─────────┴─────────┴───────┘

fig, axs = plt.subplots(
    nrows=3,
    ncols=len(df_plot),
    squeeze=False,
    figsize=(5 * len(df_plot), 12),
)
for i in range(len(df_plot)):
    axs[0][i].plot(
        np.arange(0, N) / Fs,           # Time axis [s]
        df_plot[i, 'S_t'].to_numpy(),   # Signal in time domain [-]
    )

    axs[1][i].plot(
        df_plot[i, 'F'].to_numpy(),     # Frequency axis [Hz]
        df_plot[i, 'A'].to_numpy(),     # Amplitude of the FFT [-]
    )
    axs[1][i].set_xticks(
        np.arange(0, Fs // 2 + 1, 10),  # Plot x-axis till Nyquist frequency [Hz]
    )

    axs[2][i].plot(
        df_plot[i, 'F_i'].to_numpy(),   # Normalized frequency axis [Hz]
        df_plot[i, 'A_i'].to_numpy(),   # Interpolated amplitude of the FFT [-]
    )
    axs[2][i].set_xticks(
        np.arange(0, fmax_i + 1, 1),    # Plot x-axis till maximum interpolated [Hz]
    )
plt.tight_layout()
plt.savefig(Path.cwd() / "examples" / "showcase_dsp.png")
