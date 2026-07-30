"""Showcase: signal -> Butterworth filter -> FFT -> interpolation -> feature.

Generates sine-wave signals, lowpass-filters them, computes amplitude
spectra, interpolates the spectra onto a common frequency axis, and
extracts a band-limited feature with agg_slices.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

import polars_list_utils as polist

Fs = 200.0  # Sample rate [Hz]
N = 1024    # Samples per signal (power of two)

FREQS = [i * Fs / N for i in range(N // 2 + 1)]  # FFT frequency axis [Hz]
FREQS_INTERP = list(np.linspace(0.0, 50.0, 256))  # Common axis to interpolate to


def sine_mix(freqs: list[float]) -> list[float]:
    t = np.arange(N) / Fs
    return list(sum(np.sin(2 * np.pi * f * t) for f in freqs))


df = (
    pl.DataFrame({
        "signal": [
            sine_mix([10.0]),
            sine_mix([10.0, 25.0]),
            sine_mix([10.0, 25.0, 80.0]),  # 80 Hz falls above the lowpass cutoff
        ],
    })
    # Zero-phase lowpass at 50 Hz
    .with_columns(
        polist.apply_butterworth(
            "signal",
            sample_rate=Fs,
            max_freq=50.0,
        ).alias("filtered"),
    )
    # Amplitude spectrum (a tone of amplitude A reads A at its bin)
    .with_columns(
        polist.apply_fft(
            "filtered",
            sample_rate=Fs,
            window="hann",
            scaling="amplitude",
        ).alias("fft"),
    )
    # Interpolate every spectrum onto one common frequency axis
    .with_columns(
        polist.apply_interp(
            pl.lit(FREQS),
            "fft",
            pl.lit(FREQS_INTERP),
        ).alias("fft_interp"),
    )
    # Feature: mean amplitude in the 20-30 Hz band
    .with_columns(
        polist.agg_slices(
            "fft",
            pl.lit(FREQS),
            aggregation="mean",
            slices_include=[(20.0, 30.0)],
        ).alias("mean_20_30hz"),
    )
)

print(df.select("mean_20_30hz"))

fig, axs = plt.subplots(nrows=3, ncols=len(df), squeeze=False, figsize=(5 * len(df), 9))
for i in range(len(df)):
    axs[0][i].plot(np.arange(N) / Fs, df[i, "signal"].to_numpy())
    axs[0][i].set_title(f"signal {i}")
    axs[1][i].plot(np.arange(N) / Fs, df[i, "filtered"].to_numpy())
    axs[1][i].set_title("lowpass 50 Hz (zero-phase)")
    axs[2][i].plot(FREQS_INTERP, df[i, "fft_interp"].to_numpy())
    axs[2][i].set_title("amplitude spectrum (interpolated)")
    axs[2][i].set_xlabel("Frequency [Hz]")
plt.tight_layout()
plt.savefig(Path(__file__).parent / "showcase.png")
print("Saved examples/showcase.png")
