from typing import Union
import polars as pl
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
        # 2pi because np.sin takes radians
        y += np.sin((2 * np.pi) * (x * f))
    return y


# with pl.Config(fmt_table_cell_list_len=-1, fmt_str_lengths=1000, tbl_width_chars=1000):
df = pl.DataFrame({
    'samples_original': [
        [0.0] * 1024,
        [1.0] + [0.0] * 1023,
        # [np.NAN] * 10,
        [e for e in generate_sine_wave([80, 65, 40, 10], Fs, t)[:1024]],
        [e for e in generate_sine_wave([65], Fs, t)[:1024]],
        [e for e in generate_sine_wave([40], Fs, t)[:1024]],
        [e for e in generate_sine_wave([10], Fs, t)[:1024]],
    ]
})
print(df)

df_plot1 = (
    df
    .with_columns(
        polist.apply_fft(
            'samples_original',
            sample_rate=Fs,
            # window="hanning",
            # bp_min=30,
            # bp_max=60,
            bp_ord=None,
            skip_fft=True,
        ).alias('samples_transformed'),
    )
    .with_columns(
        polist.apply_fft(
            'samples_transformed',
            sample_rate=Fs,
        ).alias('fft'),
    )
    .with_columns(
        polist.get_freqs(
            'samples_transformed',
            sample_rate=Fs,
        ).alias('freqs'),
    )
)
print(df_plot1)

# for i in range(len(df_plot1)):
#     fig, axs = plt.subplots(3, 1, squeeze=False)
#     axs[0][0].plot(
#         df_plot1[i, 'samples_original'].to_numpy(),
#     )
#     axs[1][0].plot(
#         df_plot1[i, 'samples_transformed'].to_numpy(),
#     )
#     axs[2][0].plot(
#         df_plot1[i, 'freqs'].to_numpy(),
#         df_plot1[i, 'fft'].to_numpy(),
#     )
#     plt.show()



df = pl.DataFrame({
    'samples_original': [
        [e for e in generate_sine_wave([80], Fs, t)[:1024]],
        [e for e in generate_sine_wave([65], Fs, t)[:1024]],
        [e for e in generate_sine_wave([40], Fs, t)[:1024]],
        [e for e in generate_sine_wave([10], Fs, t)[:1024]],
    ],
    # "norm_col": [80, 65, 40, 10]
    "norm_col": [40, 35, 20, 5]
})
print(df)

df_plot2 = (
    df
    .with_columns(
        polist.apply_fft(
            'samples_original',
            sample_rate=Fs,
        ).alias('fft'),
    )
    .with_columns(
        pl.col('norm_col').cast(pl.Float64)
    )
    .with_columns(
        polist.normalize_fft(
            'fft',
            norm_column='norm_col',
        ).alias('normalized_fft'),
    )
    .with_columns(
        polist.get_normalized_freqs(
            'fft',
            norm_column='norm_col',
        ).alias('freqs'),
    )
)
print(df_plot2)

for i in range(len(df_plot2)):
    fig, axs = plt.subplots(2, 1, squeeze=False)
    axs[0][0].plot(
        df_plot1[i, 'samples_original'].to_numpy(),
    )
    axs[1][0].plot(
        df_plot2[i, 'freqs'].to_numpy(),
        df_plot2[i, 'normalized_fft'].to_numpy(),
    )
    plt.show()



# WHY is the 10Hz component removed from the combined signal...
# but not from the pure 10Hz signal?

# same goes for the 65Hz component, even when max=50Hz


# check for correct scaling of the fft
# and the realpython tutorial incase of any more gotchas
# then do butterworth research
# and go through the brianmcfee dsp tutorial


