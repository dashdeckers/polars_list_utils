"""Inner-dtype preservation across every function.

Dtype preservation is an interface property, not a computation
constraint: kernels compute in f64 and round once at the exit boundary,
so Float32 results are deterministic rounded-once values. The single
exception is `count`, which emits UInt32 -- polars' count dtype --
vertically and cumulatively."""
import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist

FS = 200.0
SIGNAL32 = [float(i % 5) for i in range(32)]


def frame(inner: object) -> pl.DataFrame:
    return pl.DataFrame(
        {"s": [SIGNAL32], "i": [[float(i) for i in range(32)]]},
        schema={"s": pl.List(inner), "i": pl.List(inner)},
    )


# ------------------------------------------------------ Float32 preservation

@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
def test_transforms_preserve_the_inner_dtype(inner):
    df = frame(inner)
    fft = df.select(polist.apply_fft("s", sample_rate=FS).alias("r"))
    assert fft["r"].dtype == pl.List(inner)
    filt = df.select(
        polist.apply_butterworth("s", sample_rate=FS, max_freq=50.0).alias("r")
    )
    assert filt["r"].dtype == pl.List(inner)


def test_interp_inner_dtype_follows_y():
    # Values determine the value dtype: x and xp are coordinates.
    df = pl.DataFrame(
        {"x": [[0.0, 1.0, 2.0]], "y": [[0.0, 10.0, 20.0]]},
        schema={"x": pl.List(pl.Float64), "y": pl.List(pl.Float32)},
    )
    out = df.select(polist.apply_interp("x", "y", pl.lit([0.5, 1.5])).alias("r"))
    assert out["r"].dtype == pl.List(pl.Float32)
    assert out["r"][0].to_list() == pytest.approx([5.0, 15.0])

    # And the other way round: f32 coordinates, f64 values -> f64 out.
    df = pl.DataFrame(
        {"x": [[0.0, 1.0, 2.0]], "y": [[0.0, 10.0, 20.0]]},
        schema={"x": pl.List(pl.Float32), "y": pl.List(pl.Float64)},
    )
    out = df.select(polist.apply_interp("x", "y", pl.lit([0.5])).alias("r"))
    assert out["r"].dtype == pl.List(pl.Float64)


@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
@pytest.mark.parametrize(
    "agg", ["sum", "mean", "median", "std", "min", "max", "delta"]
)
def test_agg_slices_scalar_keeps_the_value_dtype(inner, agg):
    df = frame(inner)
    out = df.select(
        polist.agg_slices("s", "i", aggregation=agg).alias("r")
    )
    assert out["r"].dtype == inner, agg


def test_agg_slices_float32_values_are_rounded_once():
    # Computed in f64, cast at the boundary -- the value is the f64
    # answer rounded, not an f32-accumulated one.
    df = frame(pl.Float32)
    out = df.select(polist.agg_slices("s", "i", aggregation="mean").alias("r"))
    expected = sum(SIGNAL32) / len(SIGNAL32)
    assert out["r"][0] == pytest.approx(expected)


# ------------------------------------------------------------- count dtype

@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
def test_agg_slices_count_emits_uint32(inner):
    df = frame(inner)
    out = df.select(polist.agg_slices("s", "i", aggregation="count").alias("r"))
    assert out["r"].dtype == pl.UInt32
    assert out["r"][0] == 32


def test_agg_lists_count_emits_uint32():
    df = pl.DataFrame(
        {"g": [1, 1], "v": [[1.0, None], [2.0, 3.0]]},
        schema={"g": pl.Int64, "v": pl.List(pl.Float64)},
    )
    out = df.group_by("g").agg(
        polist.agg_lists("v", list_length=2, aggregation="count").alias("r")
    )
    assert out["r"].dtype == pl.List(pl.UInt32)
    assert out["r"][0].to_list() == [2, 1]


def test_both_aggregators_agree_on_count_dtype():
    # One exception to preservation, applied uniformly: UInt32 from the
    # slice reduction, the group aggregation, and the scan alike.
    horizontal = pl.DataFrame({"v": [[1.0, 2.0]], "i": [[0.0, 1.0]]}).select(
        polist.agg_slices("v", "i", aggregation="count").alias("r")
    )["r"]
    vertical = (
        pl.DataFrame({"g": [1], "v": [[1.0, 2.0]]})
        .group_by("g")
        .agg(polist.agg_lists("v", list_length=2, aggregation="count").alias("r"))["r"]
    )
    scan = pl.DataFrame({"v": [[1.0, 2.0]]}).select(
        polist.cum_agg_runs("v", pl.lit([True, True]), aggregation="count").alias("r")
    )["r"]
    assert horizontal.dtype == pl.UInt32
    assert vertical.dtype == pl.List(pl.UInt32)
    assert scan.dtype == pl.List(pl.UInt32)


# ------------------------------------------------------- integer rejection

def test_agg_lists_integer_inner_raises():
    df = pl.DataFrame(
        {"g": [1], "v": [[1, 2]]}, schema={"g": pl.Int64, "v": pl.List(pl.Int64)}
    )
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        df.group_by("g").agg(
            polist.agg_lists("v", list_length=2, aggregation="mean")
        )


@pytest.mark.parametrize(
    "build",
    [
        lambda d: d.select(polist.apply_fft("s", sample_rate=FS)),
        lambda d: d.select(
            polist.apply_butterworth("s", sample_rate=FS, max_freq=50.0)
        ),
        lambda d: d.select(polist.agg_slices("s", "s", aggregation="mean")),
    ],
)
def test_integer_rejection_lands_at_plan_time(build):
    lf = pl.LazyFrame({"s": [[1, 2, 3, 4]]}, schema={"s": pl.List(pl.Int64)})
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        build(lf).collect_schema()


# --------------------------------------------- plan-time shape checks (Array)

def test_fft_non_power_of_two_array_width_raises_at_plan_time():
    lf = pl.LazyFrame(
        {"s": [[1.0] * 6]}, schema={"s": pl.Array(pl.Float64, 6)}
    )
    with pytest.raises(polars.exceptions.PolarsError, match="power-of-two"):
        lf.select(polist.apply_fft("s", sample_rate=FS)).collect_schema()


def test_butterworth_short_array_width_raises_at_plan_time():
    # A fixed width shorter than the reflection padding means every row
    # would fail; the schema knows it, so the query never starts.
    lf = pl.LazyFrame(
        {"s": [[1.0] * 8]}, schema={"s": pl.Array(pl.Float64, 8)}
    )
    with pytest.raises(polars.exceptions.PolarsError, match="reflection padding"):
        lf.select(
            polist.apply_butterworth("s", sample_rate=FS, max_freq=50.0)
        ).collect_schema()


def test_butterworth_negative_sample_rate_raises():
    df = pl.DataFrame({"s": [SIGNAL32]})
    with pytest.raises(polars.exceptions.PolarsError, match="sample_rate"):
        df.select(polist.apply_butterworth("s", sample_rate=-1.0, max_freq=50.0))


# ---------------------------------------------------------- interp empties

def test_interp_empty_xp_returns_an_empty_list():
    # Zero query points is a valid question with an empty answer.
    df = pl.DataFrame({"x": [[0.0, 1.0]], "y": [[0.0, 10.0]]})
    out = df.select(
        polist.apply_interp(
            "x", "y", pl.lit([], dtype=pl.List(pl.Float64))
        ).alias("r")
    )
    assert out["r"][0].to_list() == []


def test_interp_empty_x_or_y_raises():
    # No interpolant at all is structural, not data.
    empty = pl.DataFrame(
        {"e": [[]], "v": [[0.0, 1.0]]},
        schema={"e": pl.List(pl.Float64), "v": pl.List(pl.Float64)},
    )
    with pytest.raises(polars.exceptions.PolarsError, match="empty"):
        empty.select(polist.apply_interp("e", "e", pl.lit([0.5])))
    with pytest.raises(polars.exceptions.PolarsError, match="empty"):
        empty.select(polist.apply_interp("v", "e", pl.lit([0.5])))
