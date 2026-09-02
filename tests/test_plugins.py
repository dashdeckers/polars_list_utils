"""AI DISCLAIMER: These tests are AI-generated, take with a grain of salt."""
import math

import numpy as np
import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist

FS = 200.0
N = 512


def sine(freq: float, amplitude: float = 1.0, n: int = N, fs: float = FS) -> list[float]:
    t = np.arange(n) / fs
    return list(amplitude * np.sin(2 * np.pi * freq * t))


def peak_bin(freq: float, n: int = N, fs: float = FS) -> int:
    return round(freq * n / fs)


# ---------------------------------------------------------------- apply_fft

def test_fft_amplitude_reads_peak_amplitude():
    # On-bin tone (25 Hz = bin 64): peak amplitude is recovered exactly.
    df = pl.DataFrame({"s": [sine(25.0, amplitude=3.0)]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, window="hann", scaling="amplitude").alias("a")
    )
    spectrum = df["a"][0].to_list()
    assert len(spectrum) == N // 2 + 1
    assert spectrum[peak_bin(25.0)] == pytest.approx(3.0, rel=1e-9)
    assert spectrum[peak_bin(80.0)] == pytest.approx(0.0, abs=1e-9)


def test_fft_hann_alias_matches_hanning():
    df = pl.DataFrame({"s": [sine(25.0)]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, window="hann", scaling="amplitude").alias("a"),
        polist.apply_fft("s", sample_rate=FS, window="hanning", scaling="amplitude").alias("b"),
    )
    assert df["a"][0].to_list() == df["b"][0].to_list()


def test_fft_spectrum_reads_mean_square():
    # scipy 'spectrum' convention: a tone of amplitude A reads A^2 / 2.
    df = pl.DataFrame({"s": [sine(25.0, amplitude=2.0)]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, scaling="spectrum").alias("p")
    )
    assert df["p"][0].to_list()[peak_bin(25.0)] == pytest.approx(2.0, rel=1e-9)


def test_fft_amplitude_squared_is_amplitude_squared():
    # Defined as amplitude^2 per bin, one-sided doubling squared along
    # with it -- the quantity "power" used to be misread as. The DC
    # offset matters: a zero-mean tone leaves the DC and Nyquist bins
    # near 1e-15, where any default absolute tolerance would accept a
    # wrong one-sided factor at exactly the two bins the doubling
    # skips.
    signal = [2.0 + v for v in sine(25.0, amplitude=3.0)]
    df = pl.DataFrame({"s": [signal]}).with_columns(
        polist.apply_fft(
            "s", sample_rate=FS, window="hann", scaling="amplitude"
        ).alias("a"),
        polist.apply_fft(
            "s", sample_rate=FS, window="hann", scaling="amplitude_squared"
        ).alias("a2"),
        polist.apply_fft(
            "s", sample_rate=FS, window="hann", scaling="spectrum"
        ).alias("sp"),
    )
    amp, amp2 = df["a"][0].to_list(), df["a2"][0].to_list()
    assert amp2 == pytest.approx([v * v for v in amp], rel=1e-12)
    assert amp2[0] == pytest.approx(amp[0] ** 2, rel=1e-12)
    assert amp2[-1] == pytest.approx(amp[-1] ** 2, rel=1e-12)
    assert amp2[peak_bin(25.0)] == pytest.approx(9.0, rel=1e-9)
    assert amp2[0] == pytest.approx(4.0, rel=1e-9)  # the DC offset, squared
    # And the documented wrinkle: spectrum == amplitude^2 / 2 at
    # interior bins, but NOT at DC, where the doubling is absent from
    # both scalings and the halving therefore has nothing to cancel.
    sp = df["sp"][0].to_list()
    assert sp[peak_bin(25.0)] == pytest.approx(amp2[peak_bin(25.0)] / 2.0, rel=1e-9)
    assert sp[0] == pytest.approx(amp2[0], rel=1e-9)


def test_fft_density_satisfies_parseval():
    # For any signal, sum(PSD) * df == mean(x^2) exactly (no window).
    rng = np.random.default_rng(42)
    signal = list(rng.standard_normal(N))
    df = pl.DataFrame({"s": [signal]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, scaling="density").alias("psd")
    )
    integral = sum(df["psd"][0].to_list()) * FS / N
    assert integral == pytest.approx(float(np.mean(np.square(signal))), rel=1e-9)


def test_fft_non_power_of_two_raises():
    # Wrong shape is structural, not data: previously a null row. The
    # escape hatch for mixed-length List data is list.len() pre-filtering.
    with pytest.raises(polars.exceptions.PolarsError, match="power-of-two"):
        pl.DataFrame({"s": [[1.0] * 100]}).with_columns(
            polist.apply_fft("s", sample_rate=FS).alias("a")
        )


def test_fft_length_one_raises():
    # A length-1 signal IS a power of two, but the windows sum to zero
    # on it; the raise is uniform across windows.
    with pytest.raises(polars.exceptions.PolarsError, match="at least 2"):
        pl.DataFrame({"s": [[5.0]]}).with_columns(
            polist.apply_fft("s", sample_rate=FS, window="hann").alias("a")
        )


def test_fft_nan_row_yields_all_nan_spectrum():
    # Invalid data propagates instead of becoming missing data: every
    # bin sums all samples, so one NaN contaminates the whole spectrum.
    signal = sine(25.0)
    signal[100] = float("nan")
    df = pl.DataFrame({"s": [signal]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, scaling="amplitude").alias("a")
    )
    spectrum = df["a"][0].to_list()
    assert spectrum is not None
    assert all(math.isnan(v) for v in spectrum)


def test_fft_inf_row_yields_nonfinite_spectrum():
    # An infinity is invalid data too, but unlike NaN it does not poison
    # every bin: inf*0 window products and inf-inf bin sums are NaN
    # while the rest stay infinite -- a visibly invalid mix, never null.
    signal = sine(25.0)
    signal[100] = float("inf")
    df = pl.DataFrame({"s": [signal]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, scaling="amplitude").alias("a")
    )
    spectrum = df["a"][0].to_list()
    assert spectrum is not None
    assert all(not math.isfinite(v) for v in spectrum)
    assert any(math.isinf(v) for v in spectrum)


def test_butterworth_nan_contaminates_not_nulls():
    # Same rule for the filter: invalid propagates, it does not vanish.
    signal = sine(5.0)
    signal[50] = float("nan")
    df = pl.DataFrame({"s": [signal]}).with_columns(
        polist.apply_butterworth("s", sample_rate=FS, max_freq=20.0).alias("f")
    )
    out = df["f"][0].to_list()
    assert out is not None
    assert any(math.isnan(v) for v in out)


def test_agg_slices_even_count_median_matches_polars():
    # The interpolated even-count median is a headline 2.0 change; pin
    # its VALUES against polars, including the extremes where the old
    # (lo + hi) / 2 midpoint overflowed or mis-signed.
    for values in (
        [1.0, 2.0, 3.0, 4.0],
        [0.1, 0.2, 0.3, 0.4],
        [1e308, 1e308],
        [-1e308, 1e308],
        [1.0, float("-inf")],
        [float("inf"), float("inf")],
    ):
        idx = [float(i) for i in range(len(values))]
        df = pl.DataFrame(
            {"v": [values], "i": [idx]},
            schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
        )
        got = df.select(
            polist.agg_slices("v", "i", aggregation="median").alias("r")
        )["r"][0]
        expected = pl.Series(values, dtype=pl.Float64).median()
        if isinstance(expected, float) and math.isnan(expected):
            assert got is not None and math.isnan(got), values
        else:
            assert got == expected, values


def test_transform_empty_list_raises():
    # The limiting case of too-short: a raise policy that softened as
    # the input got worse would be indefensible.
    df = pl.DataFrame({"s": [[]]}, schema={"s": pl.List(pl.Float64)})
    with pytest.raises(polars.exceptions.PolarsError, match="empty"):
        df.with_columns(polist.apply_fft("s", sample_rate=FS).alias("a"))
    with pytest.raises(polars.exceptions.PolarsError, match="empty"):
        df.with_columns(
            polist.apply_butterworth("s", sample_rate=FS, max_freq=50.0).alias("a")
        )


def test_fft_unknown_window_raises():
    with pytest.raises(polars.exceptions.PolarsError):
        pl.DataFrame({"s": [sine(25.0)]}).with_columns(
            polist.apply_fft(
                "s",
                sample_rate=FS,
                window="hamming",  # ty: ignore[invalid-argument-type]
            ).alias("a")
        )


@pytest.mark.parametrize("old_name", ["power", "psd"])
def test_removed_scaling_names_are_rejected(old_name):
    # The old names must be invalid, not quietly aliased -- the
    # ambiguous name disappearing is the point of the rename.
    with pytest.raises(polars.exceptions.PolarsError):
        pl.DataFrame({"s": [sine(25.0)]}).with_columns(
            polist.apply_fft(
                "s",
                sample_rate=FS,
                scaling=old_name,
            ).alias("a")
        )


def test_fft_blackman_reads_peak_amplitude():
    df = pl.DataFrame({"s": [sine(25.0, amplitude=3.0)]}).with_columns(
        polist.apply_fft("s", sample_rate=FS, window="blackman", scaling="amplitude").alias("a")
    )
    assert df["a"][0].to_list()[peak_bin(25.0)] == pytest.approx(3.0, rel=1e-9)


def test_fft_windowed_psd_enbw_identity():
    # For any signal and window, psd == power / enbw_hz elementwise;
    # the periodic hann window's ENBW is exactly 1.5 bins.
    df = pl.DataFrame({"s": [sine(25.0)]}).with_columns(
        polist.apply_fft(
            "s", sample_rate=FS, window="hann", scaling="spectrum"
        ).alias("pow"),
        polist.apply_fft(
            "s", sample_rate=FS, window="hann", scaling="density"
        ).alias("psd"),
    )
    enbw_hz = 1.5 * FS / N
    power, psd = df["pow"][0].to_list(), df["psd"][0].to_list()
    assert psd == pytest.approx([p / enbw_hz for p in power], rel=1e-12)


# -------------------------------------------------------- apply_butterworth

def test_butterworth_lowpass_removes_high_tone():
    df = pl.DataFrame(
        {"s": [list(np.array(sine(5.0)) + np.array(sine(60.0)))]}
    ).with_columns(
        polist.apply_butterworth("s", sample_rate=FS, max_freq=20.0).alias("f")
    ).with_columns(
        polist.apply_fft("f", sample_rate=FS, window="hann", scaling="amplitude").alias("a")
    )
    spectrum = df["a"][0].to_list()
    assert spectrum[peak_bin(5.0)] == pytest.approx(1.0, rel=0.05)
    assert spectrum[peak_bin(60.0)] < 1e-3


def test_butterworth_no_cutoffs_passes_through():
    df = pl.DataFrame({"s": [[1.0, 2.0, 3.0]]}).with_columns(
        polist.apply_butterworth("s", sample_rate=FS).alias("f")
    )
    assert df["f"][0].to_list() == [1.0, 2.0, 3.0]
    assert df["f"].dtype == pl.List(pl.Float64)


def test_butterworth_invalid_cutoff_raises():
    # Cutoff at/above Nyquist is a configuration error, not a null column.
    with pytest.raises(polars.exceptions.PolarsError):
        pl.DataFrame({"s": [sine(5.0)]}).with_columns(
            polist.apply_butterworth("s", sample_rate=FS, max_freq=FS / 2).alias("f")
        )
    with pytest.raises(polars.exceptions.PolarsError):
        pl.DataFrame({"s": [sine(5.0)]}).with_columns(
            polist.apply_butterworth(
                "s", sample_rate=FS, min_freq=50.0, max_freq=10.0
            ).alias("f")
        )


def test_butterworth_short_list_raises():
    # 12 samples equals the order-4 reflection padding, which once
    # panicked inside the butterworth crate and then yielded a null row;
    # wrong shape is structural, so it now raises.
    with pytest.raises(polars.exceptions.PolarsError, match="reflection padding"):
        pl.DataFrame({"s": [[1.0] * 12, sine(5.0)]}).with_columns(
            polist.apply_butterworth("s", sample_rate=FS, max_freq=20.0).alias("f")
        )


def test_butterworth_short_rows_can_be_prefiltered():
    # The documented escape hatch for mixed-length List data: filter
    # upstream of the transform. Pinned in eager and streaming alike.
    df = pl.DataFrame({"s": [[1.0] * 12, sine(5.0)]})
    out = df.filter(pl.col("s").list.len() >= 13).with_columns(
        polist.apply_butterworth("s", sample_rate=FS, max_freq=20.0).alias("f")
    )
    assert out.height == 1
    assert out["f"][0] is not None

    streamed = (
        df.lazy()
        .filter(pl.col("s").list.len() >= 13)
        .with_columns(
            polist.apply_butterworth("s", sample_rate=FS, max_freq=20.0).alias("f")
        )
        .collect(engine="streaming")
    )
    assert streamed.height == 1  # ty: ignore[unresolved-attribute]
    assert streamed["f"][0] is not None  # ty: ignore[not-subscriptable]


def test_butterworth_bandpass_short_list_raises():
    # Bandpass doubles the design order, so order 4 pads with 24 samples;
    # a 24-sample row is exactly the old panic length and must raise.
    with pytest.raises(polars.exceptions.PolarsError, match="reflection padding"):
        pl.DataFrame({"s": [[1.0] * 24, sine(25.0, n=32)]}).with_columns(
            polist.apply_butterworth(
                "s", sample_rate=FS, min_freq=10.0, max_freq=40.0
            ).alias("f")
        )


def test_butterworth_order_zero_raises():
    with pytest.raises(polars.exceptions.PolarsError):
        pl.DataFrame({"s": [sine(5.0)]}).with_columns(
            polist.apply_butterworth(
                "s", sample_rate=FS, max_freq=20.0, filter_order=0
            ).alias("f")
        )


# ------------------------------------------------------------- apply_interp

def test_interp_linear():
    df = pl.DataFrame(
        {"x": [[0.0, 1.0, 2.0]], "y": [[0.0, 10.0, 20.0]]}
    ).with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5, 1.5, 3.0])).alias("yp")
    )
    assert df["yp"][0].to_list() == pytest.approx([5.0, 15.0, 20.0])


def test_interp_length_mismatch_raises():
    with pytest.raises(polars.exceptions.PolarsError):
        pl.DataFrame(
            {"x": [[0.0, 1.0, 2.0]], "y": [[0.0, 10.0]]}
        ).with_columns(
            polist.apply_interp("x", "y", pl.lit([0.5])).alias("yp")
        )


# --------------------------------------------------------------- agg_slices

@pytest.fixture
def df_ramp() -> pl.DataFrame:
    # values 0..10 with indices 0..10
    ramp = [float(v) for v in range(11)]
    return pl.DataFrame({"v": [ramp], "i": [ramp]})


def test_agg_slices_include(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_include=[(2.0, 4.0)]
        ).alias("agg")
    )
    assert out["agg"][0] == pytest.approx(3.0)


def test_agg_slices_explicit_bounds(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="count",
            slices_include=[((2.0, "closed"), (4.0, "open"))],
        ).alias("agg")
    )
    assert out["agg"][0] == 2.0  # indices 2 and 3


def test_agg_slices_exclude_only(df_ramp):
    # Exclude without include means "everything except": 0,1 and 5..10.
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="count", slices_exclude=[(2.0, 4.0)]
        ).alias("agg")
    )
    assert out["agg"][0] == 8.0


def test_agg_slices_empty_selection_is_null(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_include=[(100.0, 200.0)]
        ).alias("mean"),
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(100.0, 200.0)]
        ).alias("cnt"),
    )
    assert out["mean"][0] is None
    assert out["cnt"][0] == 0.0


def test_agg_slices_empty_selection_sum_follows_the_empty_flag(df_ramp):
    # Summing nothing is unknown by default; "zero" restores polars'
    # identity element for callers who want the two to agree.
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=[(100.0, 200.0)]
        ).alias("null"),
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=[(100.0, 200.0)], empty="zero"
        ).alias("zero"),
    )
    assert out["null"][0] is None
    assert out["zero"][0] == 0.0
    # polars is itself split between +0.0 (Series.sum) and -0.0
    # (list.sum of an empty list); the library standardizes on +0.0.
    assert math.copysign(1.0, out["zero"][0]) == 1.0


def test_agg_slices_all_null_selection_follows_the_empty_flag():
    df = pl.DataFrame(
        {"v": [[None, None]], "i": [[0.0, 1.0]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    out = df.with_columns(
        polist.agg_slices("v", "i", aggregation="sum").alias("null"),
        polist.agg_slices("v", "i", aggregation="sum", empty="zero").alias("zero"),
    )
    assert out["null"][0] is None
    assert out["zero"][0] == 0.0


@pytest.mark.filterwarnings("ignore:In Polars 2.0")  # explode(empty_as_null)
@pytest.mark.parametrize(
    ("lo", "hi", "selects_anything"), [(0.0, 1.0, True), (100.0, 200.0, False)]
)
def test_empty_zero_reproduces_the_explode_round_trip(lo, hi, selects_anything):
    # Any list aggregation can be rewritten as explode + group_by + agg.
    # This performs that round trip rather than describing it, for both
    # a range that selects values and one that selects none -- the
    # latter is the cell where empty="zero" earns its keep.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0]], "i": [[0.0, 1.0, 2.0]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    rng: list[polist.Range] = [(lo, hi)]

    exploded = (
        df.with_row_index()
        .explode(["v", "i"])
        .group_by("index")
        .agg(
            pl.col("v")
            .filter(pl.col("i").is_between(lo, hi))
            .sum()
            .alias("r")
        )
    )["r"][0]

    got_zero = df.select(
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=rng, empty="zero"
        ).alias("r")
    )["r"][0]
    got_null = df.select(
        polist.agg_slices("v", "i", aggregation="sum", slices_include=rng).alias("r")
    )["r"][0]

    assert got_zero == exploded
    if selects_anything:
        assert got_null == exploded  # the two settings agree here
    else:
        assert got_null is None  # and deliberately part company here


@pytest.mark.parametrize("agg", ["mean", "median", "std", "min", "max", "delta", "count"])
@pytest.mark.parametrize("include", [[(0.0, 2.0)], [(100.0, 200.0)]])
def test_empty_flag_touches_only_sum(agg, include):
    # Every aggregation but sum must be byte-identical under both
    # settings, whether or not the selection came out empty.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0]], "i": [[0.0, 1.0, 2.0]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    out = df.select(
        polist.agg_slices(
            "v", "i", aggregation=agg, slices_include=include
        ).alias("null"),
        polist.agg_slices(
            "v", "i", aggregation=agg, slices_include=include, empty="zero"
        ).alias("zero"),
    )
    assert out["null"][0] == out["zero"][0]


@pytest.mark.parametrize("empty", ["null", "zero"])
def test_empty_does_not_change_a_non_empty_sum(empty):
    # Guards the `values.is_empty()` half of the kernel's condition:
    # empty="zero" must not flatten a selection that has values.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0]], "i": [[0.0, 1.0, 2.0]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    out = df.select(
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=[(0.0, 2.0)], empty=empty
        ).alias("r")
    )
    assert out["r"][0] == 6.0


def test_empty_zero_is_not_fill_null():
    # A null row and an empty selection both read null by default, but
    # they are different facts: empty="zero" resolves only the second,
    # while fill_null(0.0) cannot tell them apart and launders a
    # missing measurement into a real zero.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0], None], "i": [[0.0, 1.0], None]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    rng: list[polist.Range] = [(100.0, 200.0)]
    out = df.select(
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=rng, empty="zero"
        ).alias("zero"),
        polist.agg_slices("v", "i", aggregation="sum", slices_include=rng)
        .fill_null(0.0)
        .alias("filled"),
    )
    assert out["zero"].to_list() == [0.0, None]
    assert out["filled"].to_list() == [0.0, 0.0]


def test_empty_list_rows_are_empty_selections():
    # An empty list is the limiting case of "no index fell in range":
    # count reads 0, sum follows `empty`, everything else is null. This
    # is what makes agg_slices and agg_lists give one answer to the same
    # question, and Array(w=0) needs no special case only because it is
    # rejected outright. (Previously an empty list was a null row, which
    # neither `empty` nor count's 0 could reach.)
    df = pl.DataFrame(
        {"v": [[]], "i": [[]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    out = df.select(
        polist.agg_slices("v", "i", aggregation="sum").alias("sum_null"),
        polist.agg_slices("v", "i", aggregation="sum", empty="zero").alias("sum_zero"),
        polist.agg_slices("v", "i", aggregation="count").alias("count"),
        polist.agg_slices("v", "i", aggregation="mean").alias("mean"),
    )
    assert out["sum_null"][0] is None
    assert out["sum_zero"][0] == 0.0
    assert out["count"][0] == 0
    assert out["mean"][0] is None


@pytest.mark.parametrize("bad", ["nul", "Null", "zeroes", None, 0, ""])
def test_both_aggregators_reject_a_bogus_empty(bad):
    # Falling through to "zero" on a typo would silently pick the very
    # reading the flag exists to avoid.
    with pytest.raises(ValueError, match="empty must be"):
        polist.agg_slices("v", "i", aggregation="sum", empty=bad)
    with pytest.raises(ValueError, match="empty must be"):
        polist.agg_lists("v", list_length=1, aggregation="sum", empty=bad)


def test_both_aggregators_reject_a_bogus_aggregation():
    with pytest.raises(ValueError, match="aggregation must be"):
        polist.agg_slices("v", "i", aggregation="total")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="aggregation must be"):
        polist.agg_lists("v", list_length=1, aggregation="total")  # ty: ignore[invalid-argument-type]


def test_agg_slices_matches_polars_vertical_semantics():
    # The contract: aggregating a slice behaves exactly like polars'
    # vertical aggregations over the same values (nulls skipped, NaN
    # propagates per-kernel, ddof=1 std).
    data = [1.0, None, float("nan"), 3.0]
    s = pl.Series(data, dtype=pl.Float64)
    aggs: list[polist.Aggregation] = ["sum", "mean", "median", "min", "max", "std", "count"]
    expected = {
        "sum": s.sum(),
        "mean": s.mean(),
        "median": s.median(),
        "min": s.min(),
        "max": s.max(),
        "std": s.std(),
        "count": float(s.count()),
    }
    df = pl.DataFrame({"v": [data]}, schema={"v": pl.List(pl.Float64)})
    out = df.with_columns(
        polist.agg_slices(
            "v", pl.lit([0.0, 1.0, 2.0, 3.0]), aggregation=agg
        ).alias(agg)
        for agg in aggs
    )
    for agg in aggs:
        exp = expected[agg]
        got = out[agg][0]
        if exp is None:
            assert got is None, agg
        elif isinstance(exp, float) and math.isnan(exp):
            assert math.isnan(got), agg
        else:
            assert got == pytest.approx(exp), agg


def test_agg_slices_nulls_skipped_pairwise():
    # A null in either list drops the pair, not the row.
    df = pl.DataFrame(
        {
            "v": [[1.0, None, 3.0], [1.0, 2.0, 3.0]],
            "i": [[0.0, 1.0, 2.0], [0.0, None, 2.0]],
        }
    )
    out = df.with_columns(
        polist.agg_slices("v", "i", aggregation="mean").alias("mean"),
        polist.agg_slices("v", "i", aggregation="count").alias("cnt"),
    )
    assert out["mean"].to_list() == pytest.approx([2.0, 2.0])
    assert out["cnt"].to_list() == [2.0, 2.0]


def test_agg_slices_all_nan_matches_polars():
    # All-NaN selections read NaN for min/max (not +-inf fold seeds).
    df = pl.DataFrame({"v": [[float("nan")] * 2], "i": [[0.0, 1.0]]})
    out = df.with_columns(
        polist.agg_slices("v", "i", aggregation=agg).alias(agg) # ty: ignore[invalid-argument-type]
        for agg in ["sum", "mean", "median", "min", "max", "delta"]
    )
    for agg in ["sum", "mean", "median", "min", "max", "delta"]:
        assert math.isnan(out[agg][0]), agg


def test_agg_slices_nan_indices_never_match():
    df = pl.DataFrame({"v": [[1.0, 2.0, 3.0]], "i": [[0.0, float("nan"), 2.0]]})
    out = df.with_columns(
        polist.agg_slices("v", "i", aggregation="count").alias("cnt")
    )
    assert out["cnt"][0] == 2.0


def test_agg_slices_length_mismatch_raises(df_ramp):
    with pytest.raises(polars.exceptions.PolarsError):
        df_ramp.with_columns(
            polist.agg_slices("v", pl.lit([0.0, 1.0]), aggregation="mean").alias("agg")
        )


def test_agg_slices_aggregations(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices("v", "i", aggregation=agg).alias(agg) # ty: ignore[invalid-argument-type]
        for agg in ["sum", "mean", "median", "min", "max", "std", "delta", "count"]
    )
    assert out["sum"][0] == pytest.approx(55.0)
    assert out["mean"][0] == pytest.approx(5.0)
    assert out["median"][0] == pytest.approx(5.0)
    assert out["min"][0] == 0.0
    assert out["max"][0] == 10.0
    assert out["std"][0] == pytest.approx(float(np.std(np.arange(11), ddof=1)))
    assert out["delta"][0] == 10.0
    assert out["count"][0] == 11.0


# ---------------------------------------------------------------- agg_lists

def test_agg_lists_elementwise_over_groups():
    df = pl.DataFrame(
        {
            "g": [1, 1, 2],
            "v": [[1.0, None, float("nan")], [3.0, 4.0, 5.0], [7.0, 8.0, 9.0]],
        }
    )
    out = (
        df.group_by("g")
        .agg(
            polist.agg_lists("v", list_length=3, aggregation="mean").alias("mean"),
            polist.agg_lists("v", list_length=3, aggregation="count").alias("cnt"),
            polist.agg_lists("v", list_length=3, aggregation="delta").alias("delta"),
        )
        .sort("g")
    )
    mean = out["mean"][0].to_list()
    assert mean[0] == pytest.approx(2.0)  # (1 + 3) / 2
    assert mean[1] == pytest.approx(4.0)  # null skipped from numerator and denominator
    assert math.isnan(mean[2])  # NaN poisons the mean
    assert out["cnt"][0].to_list() == [2.0, 1.0, 2.0]  # count skips nulls
    assert out["delta"][0].to_list() == [2.0, 0.0, 0.0]  # min/max skip NaN
    assert out["mean"][1].to_list() == pytest.approx([7.0, 8.0, 9.0])


def test_agg_lists_sum_of_all_null_position_follows_the_empty_flag():
    # The same switch as agg_slices, so both aggregating functions give
    # one answer to "what is the sum of no values".
    df = pl.DataFrame(
        {"g": [1, 1], "v": [[1.0, None], [2.0, None]]},
        schema={"g": pl.Int64, "v": pl.List(pl.Float64)},
    )
    out = df.group_by("g").agg(
        polist.agg_lists("v", list_length=2, aggregation="sum").alias("null"),
        polist.agg_lists(
            "v", list_length=2, aggregation="sum", empty="zero"
        ).alias("zero"),
    )
    assert out["null"][0].to_list() == [3.0, None]
    assert out["zero"][0].to_list() == [3.0, 0.0]


def test_agg_lists_invalid_length_raises():
    with pytest.raises(ValueError, match="list_length"):
        polist.agg_lists("v", list_length=0, aggregation="mean")


@pytest.mark.parametrize("empty", ["null", "zero"])
def test_both_aggregators_agree_on_the_sum_of_nothing(empty):
    # The two functions must give one answer to "what is the sum of no
    # values", under either setting.
    horizontal = pl.DataFrame(
        {"v": [[None, None]], "i": [[0.0, 1.0]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    ).select(
        polist.agg_slices("v", "i", aggregation="sum", empty=empty).alias("r")
    )["r"][0]
    vertical = (
        pl.DataFrame(
            {"g": [1, 1], "v": [[None], [None]]},
            schema={"g": pl.Int64, "v": pl.List(pl.Float64)},
        )
        .group_by("g")
        .agg(
            polist.agg_lists(
                "v", list_length=1, aggregation="sum", empty=empty
            ).alias("r")
        )["r"][0]
        .to_list()[0]
    )
    assert horizontal == vertical


def test_agg_lists_and_agg_slices_share_missing_data_semantics():
    # The same values with a null and a NaN, aggregated horizontally by
    # agg_slices and vertically by agg_lists, must agree for every
    # aggregation.
    data = [1.0, None, float("nan"), 3.0]
    df_h = pl.DataFrame({"v": [data]}, schema={"v": pl.List(pl.Float64)})
    df_v = pl.DataFrame(
        {"g": [1] * 4, "v": [[x] for x in data]},
        schema={"g": pl.Int64, "v": pl.List(pl.Float64)},
    )
    for agg in ["sum", "mean", "median", "std", "min", "max", "delta", "count"]:
        h = df_h.with_columns(
            polist.agg_slices(
                "v", pl.lit([0.0, 1.0, 2.0, 3.0]), aggregation=agg # ty: ignore[invalid-argument-type]
            ).alias("r")
        )["r"][0]
        v = df_v.group_by("g").agg(
            polist.agg_lists("v", list_length=1, aggregation=agg).alias("r") # ty: ignore[invalid-argument-type]
        )["r"][0].to_list()[0]
        if h is None or v is None:
            assert h == v, agg
        elif math.isnan(h) or math.isnan(v):
            assert math.isnan(h) and math.isnan(v), agg
        else:
            assert h == pytest.approx(v), agg


# ----------------------------------------------------------- null semantics

NULL_ELEMENT_TRANSFORMS = {
    # Every null-element position across the transform family: a single
    # entry-path change (dropping nulls, unwrap_or) must fail loudly.
    "fft": lambda good, bad: pl.DataFrame({"s": [good, bad]}).select(
        polist.apply_fft("s", sample_rate=FS).alias("r")
    ),
    "butterworth": lambda good, bad: pl.DataFrame({"s": [good, bad]}).select(
        polist.apply_butterworth("s", sample_rate=FS, max_freq=20.0).alias("r")
    ),
    # The no-cutoff pass-through is not a no-op: the family entry
    # contract still applies (spec 9.20).
    "butterworth_passthrough": lambda good, bad: pl.DataFrame(
        {"s": [good, bad]}
    ).select(polist.apply_butterworth("s", sample_rate=FS).alias("r")),
    "interp_x": lambda good, bad: pl.DataFrame(
        {"x": [good, bad], "y": [good, good]}
    ).select(polist.apply_interp("x", "y", pl.lit([0.5]), strict=False).alias("r")),
    "interp_y": lambda good, bad: pl.DataFrame(
        {"x": [good, good], "y": [good, bad]}
    ).select(polist.apply_interp("x", "y", pl.lit([0.5])).alias("r")),
    "interp_xp": lambda good, bad: pl.DataFrame(
        {"x": [good, good], "y": [good, good], "p": [good, bad]}
    ).select(polist.apply_interp("x", "y", "p").alias("r")),
}


@pytest.mark.parametrize("position", list(NULL_ELEMENT_TRANSFORMS))
def test_null_elements_null_the_row_across_the_family(position):
    good = [float(v) for v in range(16)]
    bad = [0.0, None] + [float(v) for v in range(14)]
    out = NULL_ELEMENT_TRANSFORMS[position](good, bad)["r"]
    assert out[0] is not None, position
    assert out[1] is None, position  # inner nulls must not silently misalign


def test_null_and_inner_null_rows_yield_null():
    df = pl.DataFrame(
        {"s": [sine(25.0), None, [1.0, None, 3.0] + [0.0] * 509]},
        schema={"s": pl.List(pl.Float64)},
    ).with_columns(
        polist.apply_fft("s", sample_rate=FS, scaling="amplitude").alias("a")
    )
    assert df["a"][0] is not None
    assert df["a"][1] is None
    assert df["a"][2] is None  # inner nulls must not silently misalign


def test_integer_lists_raise():
    # Previously accepted and silently cast to f64; integer support is
    # deliberately out of scope, so the rejection is loud, at plan time,
    # and names the parameter.
    df = pl.DataFrame({"x": [[0, 1, 2]], "y": [[0, 10, 20]]})
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        df.with_columns(polist.apply_interp("x", "y", pl.lit([1.5])).alias("yp"))
    lf = df.lazy().with_columns(
        polist.apply_interp("x", "y", pl.lit([1.5])).alias("yp")
    )
    with pytest.raises(polars.exceptions.PolarsError, match="x_column"):
        lf.collect_schema()


def test_array_input_is_supported_not_aborting():
    # Without the polars `dtype-array` feature, an Array input panicked in
    # polars' arrow-FFI import before any plugin check ran -- a
    # non-unwinding panic that aborted the whole process. It is now a
    # supported container; see tests/test_array_container.py for the
    # container algebra. This keeps the original regression honest: the
    # calls that used to abort must complete.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0], [3.0, 4.0]]},
        schema={"v": pl.Array(pl.Float64, 2)},
    )
    out = df.with_columns(
        polist.agg_slices("v", "v", aggregation="mean").alias("a"),
        polist.apply_fft("v", sample_rate=FS).alias("f"),
    )
    assert out["a"].to_list() == pytest.approx([1.5, 3.5])
    assert out["f"].dtype == pl.Array(pl.Float64, 2)


def test_broadcasting_literal_over_rows():
    df = pl.DataFrame({"v": [[1.0, 2.0], [3.0, 4.0]]}).with_columns(
        polist.agg_slices("v", pl.lit([0.0, 1.0]), aggregation="mean").alias("agg")
    )
    assert df["agg"].to_list() == pytest.approx([1.5, 3.5])


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_lazy_execution_matches_eager(engine):
    # The plugin must survive lazy/streaming execution, where it runs
    # per-morsel and re-broadcasts the literal for every batch.
    df = pl.DataFrame({"v": [[float(i), float(i) + 1.0] for i in range(2000)]})
    expr = polist.agg_slices("v", pl.lit([0.0, 1.0]), aggregation="mean").alias("agg")
    eager = df.with_columns(expr)["agg"].to_list()
    lazy = df.lazy().with_columns(expr).collect(engine=engine).select("agg").to_series().to_list() # ty: ignore[unresolved-attribute]
    assert lazy == pytest.approx(eager)
