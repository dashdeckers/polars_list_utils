"""Array support.

The kernels only ever see List: Array is normalized on the way in and
rebuilt on the way out. So the oracle for every value question is the
List result, and what needs testing on top is the container algebra --
which container comes out, and which structural mismatches the schema
can catch before execution."""
import polars as pl
import polars.exceptions
import pytest
from polars.datatypes import DataTypeClass

import polars_list_utils as polist

FS = 200.0
# A power of two for the FFT, and longer than the order-4 Butterworth
# reflection padding (3 * 4 + 1) so the filter produces a real row.
SIGNAL = [float(i % 5) for i in range(32)]


def as_array(df: pl.DataFrame, **cols: DataTypeClass) -> pl.DataFrame:
    """The same frame with the named List columns retyped as Arrays,
    each at the width of its first row."""
    return df.with_columns(
        pl.col(name).cast(pl.Array(inner, len(df[name][0])))
        for name, inner in cols.items()
    )


# ------------------------------------------------- values are container-blind

def test_transforms_agree_between_containers():
    lists = pl.DataFrame({"s": [SIGNAL]}, schema={"s": pl.List(pl.Float64)})
    arrays = as_array(lists, s=pl.Float64)
    for expr in [
        polist.apply_fft("s", sample_rate=FS, window="hanning", scaling="amplitude"),
        polist.apply_butterworth("s", sample_rate=FS, max_freq=50.0),
    ]:
        got = arrays.select(expr.alias("r"))["r"][0].to_list()
        expected = lists.select(expr.alias("r"))["r"][0].to_list()
        assert got == pytest.approx(expected)


def test_aggregations_agree_between_containers():
    lists = pl.DataFrame(
        {"v": [[1.0, None, 3.0, 4.0]], "i": [[0.0, 1.0, 2.0, 3.0]]},
        schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
    )
    arrays = as_array(lists, v=pl.Float64, i=pl.Float64)
    for agg in ["sum", "mean", "median", "std", "min", "max", "delta", "count"]:
        expr = polist.agg_slices("v", "i", aggregation=agg)  # ty: ignore[invalid-argument-type]
        assert arrays.select(expr.alias("r"))["r"][0] == pytest.approx(
            lists.select(expr.alias("r"))["r"][0]
        ), agg


def test_elementwise_and_scan_agree_between_containers():
    lists = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0, 4.0]], "g": [[True, True, False, True]]},
        schema={"v": pl.List(pl.Float64), "g": pl.List(pl.Boolean)},
    )
    arrays = as_array(lists, v=pl.Float64, g=pl.Boolean)
    for expr in [
        polist.zip_binary("v", "v", op="mul"),
        polist.cum_agg_runs("v", "g", aggregation="sum"),
        polist.cum_agg_runs("v", "g", aggregation="count"),
    ]:
        assert (
            arrays.select(expr.alias("r"))["r"][0].to_list()
            == lists.select(expr.alias("r"))["r"][0].to_list()
        )


# ------------------------------------------------------ container propagation

def test_length_preserving_transforms_keep_the_container():
    df = as_array(
        pl.DataFrame({"s": [SIGNAL]}, schema={"s": pl.List(pl.Float64)}), s=pl.Float64
    )
    out = df.select(
        polist.apply_butterworth("s", sample_rate=FS, max_freq=50.0).alias("r")
    )
    assert out["r"].dtype == pl.Array(pl.Float64, len(SIGNAL))
    assert len(out["r"][0].to_list()) == len(SIGNAL)


def test_fft_halves_the_array_width():
    # The one width-transforming function: n samples -> n/2 + 1 bins.
    df = as_array(
        pl.DataFrame({"s": [SIGNAL]}, schema={"s": pl.List(pl.Float64)}), s=pl.Float64
    )
    half = len(SIGNAL) // 2 + 1
    out = df.select(polist.apply_fft("s", sample_rate=FS).alias("r"))
    assert out["r"].dtype == pl.Array(pl.Float64, half)
    assert len(out["r"][0].to_list()) == half


def test_interp_container_follows_xp():
    # The output holds one value per query coordinate, so xp decides the
    # shape and the x/y containers are irrelevant.
    df = pl.DataFrame(
        {"x": [[0.0, 1.0, 2.0]], "y": [[0.0, 10.0, 20.0]]},
        schema={"x": pl.List(pl.Float64), "y": pl.List(pl.Float64)},
    )
    xp_array = pl.Series("xp", [[0.5, 1.5]], dtype=pl.Array(pl.Float64, 2))
    out = df.select(polist.apply_interp("x", "y", pl.lit(xp_array)).alias("r"))
    assert out["r"].dtype == pl.Array(pl.Float64, 2)
    assert out["r"][0].to_list() == pytest.approx([5.0, 15.0])

    # List xp over Array x/y gives List back.
    arrays = as_array(df, x=pl.Float64, y=pl.Float64)
    out = arrays.select(polist.apply_interp("x", "y", pl.lit([0.5])).alias("r"))
    assert out["r"].dtype == pl.List(pl.Float64)


def test_agg_slices_is_scalar_regardless_of_container():
    df = as_array(
        pl.DataFrame({"v": [[1.0, 2.0]], "i": [[0.0, 1.0]]},
                     schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)}),
        v=pl.Float64, i=pl.Float64,
    )
    out = df.select(polist.agg_slices("v", "i", aggregation="mean").alias("r"))
    assert out["r"].dtype == pl.Float64


def test_agg_lists_returns_list_even_for_array_input():
    # Documented exception to "output container follows input": agg_lists
    # is an expression composition, not a plugin -- it reduces rows, which
    # an elementwise plugin cannot -- so it never sees the input schema at
    # the point where the output container would have to be chosen.
    df = pl.DataFrame(
        {"g": [1, 1], "a": [[1.0, 2.0], [3.0, 4.0]]},
        schema={"g": pl.Int64, "a": pl.Array(pl.Float64, 2)},
    )
    out = df.group_by("g").agg(
        polist.agg_lists("a", list_length=2, aggregation="mean").alias("r")
    )
    assert out["r"].dtype == pl.List(pl.Float64)
    assert out["r"][0].to_list() == pytest.approx([2.0, 3.0])


@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
def test_inner_dtype_survives_the_container_round_trip(inner):
    df = pl.DataFrame(
        {"v": [[1.0, 2.0]], "g": [[True, True]]},
        schema={"v": pl.Array(inner, 2), "g": pl.Array(pl.Boolean, 2)},
    )
    assert df.select(polist.zip_binary("v", "v", op="add").alias("r"))[
        "r"
    ].dtype == pl.Array(inner, 2)
    assert df.select(
        polist.cum_agg_runs("v", "g", aggregation="sum").alias("r")
    )["r"].dtype == pl.Array(inner, 2)
    # count keeps polars' count dtype, in the input's container.
    assert df.select(
        polist.cum_agg_runs("v", "g", aggregation="count").alias("r")
    )["r"].dtype == pl.Array(pl.UInt32, 2)


def test_comparison_on_arrays_yields_boolean_array():
    df = pl.DataFrame({"a": [[1.0, 2.0]]}, schema={"a": pl.Array(pl.Float64, 2)})
    out = df.select(polist.zip_binary("a", "a", op="le").alias("r"))
    assert out["r"].dtype == pl.Array(pl.Boolean, 2)


# --------------------------------------------------------- mixed and mismatched

def test_mixed_containers_yield_a_list():
    df = pl.DataFrame(
        {"a": [[1.0, 2.0]], "b": [[3.0, 4.0]]},
        schema={"a": pl.Array(pl.Float64, 2), "b": pl.List(pl.Float64)},
    )
    out = df.select(polist.zip_binary("a", "b", op="add").alias("r"))
    assert out["r"].dtype == pl.List(pl.Float64)
    assert out["r"][0].to_list() == [4.0, 6.0]


def test_array_width_mismatch_raises_at_plan_time():
    df = pl.LazyFrame(
        {"a": [[1.0, 2.0]], "b": [[1.0, 2.0, 3.0]]},
        schema={"a": pl.Array(pl.Float64, 2), "b": pl.Array(pl.Float64, 3)},
    )
    with pytest.raises(polars.exceptions.PolarsError, match="equal widths"):
        df.select(polist.zip_binary("a", "b", op="add")).collect_schema()


def test_cum_agg_runs_array_width_mismatch_raises_at_plan_time():
    df = pl.LazyFrame(
        {"v": [[1.0, 2.0]], "g": [[True, True, True]]},
        schema={"v": pl.Array(pl.Float64, 2), "g": pl.Array(pl.Boolean, 3)},
    )
    with pytest.raises(polars.exceptions.PolarsError, match="equal widths"):
        df.select(polist.cum_agg_runs("v", "g", aggregation="sum")).collect_schema()


def test_agg_lists_validates_list_length_against_array_width():
    df = pl.DataFrame(
        {"g": [1], "a": [[1.0, 2.0]]},
        schema={"g": pl.Int64, "a": pl.Array(pl.Float64, 2)},
    )
    with pytest.raises(polars.exceptions.PolarsError, match="does not match the Array width"):
        df.group_by("g").agg(polist.agg_lists("a", list_length=3, aggregation="mean"))


def test_integer_arrays_raise_for_the_dtype_strict_functions():
    df = pl.DataFrame({"a": [[1, 2]]}, schema={"a": pl.Array(pl.Int64, 2)})
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        df.select(polist.zip_binary("a", "a", op="add"))


# ------------------------------------------------------------- zero-width edge

@pytest.mark.parametrize(
    "build",
    [
        lambda d: d.select(polist.agg_slices("a", "a", aggregation="count")),
        lambda d: d.select(polist.zip_binary("a", "a", op="add")),
        lambda d: d.select(polist.apply_fft("a", sample_rate=FS)),
    ],
)
def test_zero_width_arrays_raise_instead_of_panicking(build):
    # polars-arrow panics when slicing a zero-width FixedSizeList, inside
    # the FFI boundary before any of this library's code runs. Every
    # function resolves its output type at plan time, so the query is
    # stopped there -- a typed error, never a panic. A zero-length List
    # stays perfectly valid.
    df = pl.DataFrame({"a": [[]]}, schema={"a": pl.Array(pl.Float64, 0)})
    with pytest.raises(polars.exceptions.PolarsError) as excinfo:
        build(df)
    message = str(excinfo.value)
    assert "zero-width Arrays are not supported" in message
    # "the plugin panicked" is how pyo3-polars reports an unwound panic.
    assert "the plugin panicked" not in message


def test_zero_length_lists_are_unaffected():
    df = pl.DataFrame({"a": [[]]}, schema={"a": pl.List(pl.Float64)})
    out = df.select(polist.zip_binary("a", "a", op="add").alias("r"))
    assert out["r"][0].to_list() == []


# ------------------------------------------------------- nulls and broadcasting

def test_null_rows_and_inner_nulls_behave_as_for_lists():
    df = pl.DataFrame(
        {"a": [[1.0, 2.0], None]}, schema={"a": pl.Array(pl.Float64, 2)}
    )
    out = df.select(polist.zip_binary("a", "a", op="add").alias("r"))
    assert out["r"].to_list() == [[2.0, 4.0], None]

    inner_null = pl.DataFrame(
        {"a": [[1.0, None]], "i": [[0.0, 1.0]]},
        schema={"a": pl.Array(pl.Float64, 2), "i": pl.Array(pl.Float64, 2)},
    )
    assert inner_null.select(
        polist.agg_slices("a", "i", aggregation="count").alias("r")
    )["r"][0] == 1.0


def test_array_literal_broadcasts():
    df = pl.DataFrame(
        {"a": [[1.0, 2.0], [3.0, 4.0]]}, schema={"a": pl.Array(pl.Float64, 2)}
    )
    template = pl.Series("t", [[10.0, 20.0]], dtype=pl.Array(pl.Float64, 2))
    out = df.select(polist.zip_binary("a", pl.lit(template), op="add").alias("r"))
    assert out["r"].dtype == pl.Array(pl.Float64, 2)
    assert out["r"].to_list() == [[11.0, 22.0], [13.0, 24.0]]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_arrays_survive_lazy_and_streaming(engine):
    df = pl.DataFrame(
        {"a": [[float(i), float(i) + 1.0] for i in range(2000)]},
        schema={"a": pl.Array(pl.Float64, 2)},
    )
    expr = polist.zip_binary("a", "a", op="add").alias("r")
    eager = df.select(expr)["r"].to_list()
    lazy = df.lazy().select(expr).collect(engine=engine)["r"].to_list()  # ty: ignore[not-subscriptable]
    assert lazy == eager
