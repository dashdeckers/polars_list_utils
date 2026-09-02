"""Every strict check gets a positive test (raises under strict=True)
and a negative test (the documented silent degenerate result under the
default)."""
import math

import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist

NAN = float("nan")


# ------------------------------------------------------------ apply_interp

def _interp_frame() -> pl.DataFrame:
    # Unsorted x: the sorted answer would be [5.0, 15.0].
    return pl.DataFrame({"x": [[2.0, 0.0, 1.0]], "y": [[20.0, 0.0, 10.0]]})


def test_interp_strict_raises_on_unsorted_x():
    with pytest.raises(polars.exceptions.PolarsError, match="non-decreasing"):
        _interp_frame().with_columns(
            polist.apply_interp("x", "y", pl.lit([0.5, 1.5]), strict=True).alias("r")
        )


def test_strict_is_the_default():
    # Feature-extraction-friendly defaults: the silent failures are
    # errors unless the caller opts out.
    with pytest.raises(polars.exceptions.PolarsError, match="non-decreasing"):
        _interp_frame().with_columns(
            polist.apply_interp("x", "y", pl.lit([0.5, 1.5])).alias("r")
        )
    with pytest.raises(ValueError, match="lo <= hi"):
        polist.agg_slices("v", "i", aggregation="mean", slices_include=[(2.0, 0.0)])
    with pytest.raises(polars.exceptions.PolarsError, match="list_length"):
        pl.DataFrame({"g": [1], "v": [[1.0, 2.0, 3.0]]}).group_by("g").agg(
            polist.agg_lists("v", list_length=2, aggregation="mean")
        )


def test_interp_lenient_silently_interpolates_garbage():
    out = _interp_frame().with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5, 1.5]), strict=False).alias("r")
    )
    assert out["r"][0].to_list() == [20.0, 20.0]


@pytest.mark.parametrize("strict", [False, True])
def test_interp_nan_x_propagates_rather_than_raising(strict):
    # NaN is a legitimate float for the transform family: it flows into
    # the output instead of raising. The strict check is for unsorted x,
    # which fails silently; a NaN announces itself in the result.
    df = pl.DataFrame({"x": [[0.0, NAN, 2.0]], "y": [[0.0, 1.0, 2.0]]})
    out = df.with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5, 1.5]), strict=strict).alias("r")
    )
    assert all(math.isnan(v) for v in out["r"][0].to_list())


def test_interp_strict_ignores_lone_nan_x():
    # A length-1 x has no adjacent pair; NaN is not a violation anyway.
    df = pl.DataFrame({"x": [[NAN]], "y": [[1.0]]})
    out = df.with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5]), strict=True).alias("r")
    )
    assert out["r"][0].to_list() == [1.0]


def test_interp_strict_sees_a_descent_masked_by_nan():
    # Every IEEE comparison with NaN is false, so an adjacent-pair test
    # reads [0, 10, NaN, 1, 2] as sorted and silently interpolates
    # finite, plausible, wrong numbers -- exactly the failure the check
    # exists for.
    df = pl.DataFrame(
        {"x": [[0.0, 10.0, NAN, 1.0, 2.0]], "y": [[0.0, 1.0, 2.0, 3.0, 4.0]]}
    )
    with pytest.raises(polars.exceptions.PolarsError, match="non-decreasing"):
        df.with_columns(
            polist.apply_interp("x", "y", pl.lit([0.5]), strict=True).alias("r")
        )


def test_interp_strict_allows_nan_in_genuinely_sorted_x():
    # NaN itself is not a violation; it propagates into the output.
    df = pl.DataFrame({"x": [[0.0, NAN, 2.0]], "y": [[0.0, 1.0, 2.0]]})
    out = df.with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5]), strict=True).alias("r")
    )
    assert math.isnan(out["r"][0].to_list()[0])


def test_interp_strict_accepts_duplicate_x():
    # Duplicates stay legal, as in numpy.
    df = pl.DataFrame({"x": [[0.0, 1.0, 1.0, 2.0]], "y": [[0.0, 10.0, 20.0, 30.0]]})
    out = df.with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5]), strict=True).alias("r")
    )
    assert out["r"][0].to_list() == pytest.approx([5.0])


# -------------------------------------------------------------- agg_slices

def test_agg_slices_strict_raises_on_inverted_range():
    # A construction-time check: no DataFrame is needed to trigger it.
    with pytest.raises(ValueError, match="lo <= hi"):
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_include=[(2.0, 0.0)], strict=True
        )


def test_agg_slices_strict_raises_on_nan_bound():
    with pytest.raises(ValueError, match="non-NaN"):
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_exclude=[(NAN, 1.0)], strict=True
        )
    with pytest.raises(ValueError, match="non-NaN"):
        polist.agg_slices(
            "v",
            "i",
            aggregation="mean",
            slices_include=[((0.0, "closed"), (NAN, "open"))],
            strict=True,
        )


def test_agg_slices_lenient_nan_bound_is_empty_selection():
    # The negative half: with strict off, a NaN bound is a legitimately
    # empty selection rather than an error.
    df = pl.DataFrame({"v": [[1.0, 2.0]], "i": [[0.0, 1.0]]})
    out = df.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_include=[(NAN, 1.0)], strict=False
        ).alias("mean"),
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(NAN, 1.0)], strict=False
        ).alias("cnt"),
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=[(NAN, 1.0)], strict=False
        ).alias("sum"),
    )
    assert out["mean"][0] is None
    assert out["cnt"][0] == 0.0
    assert out["sum"][0] is None


def test_agg_slices_strict_accepts_list_shaped_ranges():
    # Ranges loaded from JSON arrive as lists, which the plugin accepts;
    # strict must validate them, not reject them.
    df = pl.DataFrame({"v": [[1.0, 2.0, 3.0]], "i": [[0.0, 1.0, 2.0]]})
    out = df.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="sum",
            slices_include=[[[0.0, "closed"], [1.0, "closed"]]],  # ty: ignore[invalid-argument-type]
            strict=True,
        ).alias("r")
    )
    assert out["r"][0] == 3.0
    with pytest.raises(ValueError, match="lo <= hi"):
        polist.agg_slices(
            "v", "i",
            aggregation="sum",
            slices_include=[[[2.0, "closed"], [1.0, "closed"]]],  # ty: ignore[invalid-argument-type]
            strict=True,
        )


def test_agg_slices_lenient_inverted_range_is_empty_selection():
    df = pl.DataFrame({"v": [[1.0, 2.0]], "i": [[0.0, 1.0]]})
    out = df.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_include=[(2.0, 0.0)], strict=False
        ).alias("mean"),
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(2.0, 0.0)], strict=False
        ).alias("cnt"),
    )
    assert out["mean"][0] is None
    assert out["cnt"][0] == 0.0


# --------------------------------------------------------------- agg_lists

def test_agg_lists_strict_raises_on_overlong_list():
    df = pl.DataFrame({"g": [1, 1], "v": [[1.0, 2.0, 3.0], [1.0, 2.0]]})
    with pytest.raises(polars.exceptions.PolarsError, match="list_length"):
        df.group_by("g").agg(
            polist.agg_lists("v", list_length=2, aggregation="mean", strict=True)
        )


def test_agg_lists_strict_allows_null_padded_rows():
    # Truncating a null-padded tail discards nothing -- every
    # aggregation skips nulls anyway -- so raising would be a false
    # positive whose only remedy is disabling the check outright.
    # Fixed-capacity buffers and widened ragged data look like this.
    df = pl.DataFrame(
        {"g": [1, 1], "v": [[1.0, 2.0, None, None], [3.0, 4.0, None, None]]},
        schema={"g": pl.Int64, "v": pl.List(pl.Float64)},
    )
    out = df.group_by("g").agg(
        polist.agg_lists("v", list_length=2, aggregation="mean", strict=True).alias("r")
    )
    assert out["r"][0].to_list() == pytest.approx([2.0, 3.0])


def test_agg_lists_lenient_truncates_silently():
    # list_length as a deliberate window: third elements are dropped.
    df = pl.DataFrame({"g": [1, 1], "v": [[1.0, 2.0, 30.0], [3.0, 4.0, 50.0]]})
    out = df.group_by("g").agg(
        polist.agg_lists(
            "v", list_length=2, aggregation="mean", strict=False
        ).alias("r")
    )
    assert out["r"][0].to_list() == pytest.approx([2.0, 3.0])


def test_agg_lists_strict_passes_within_length():
    # Shorter lists stay legal under strict: they contribute nulls at
    # the missing positions, which is data, not a violation. So are null
    # rows, which pass through the check untouched.
    df = pl.DataFrame(
        {"g": [1, 1, 1], "v": [[1.0, 2.0], [3.0], None]},
        schema={"g": pl.Int64, "v": pl.List(pl.Float64)},
    )
    out = df.group_by("g").agg(
        polist.agg_lists("v", list_length=2, aggregation="mean", strict=True).alias("r")
    )
    assert out["r"][0].to_list() == pytest.approx([2.0, 2.0])
