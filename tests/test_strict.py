"""Every strict check gets a positive test (raises under strict=True)
and a negative test (the documented silent degenerate result under the
default)."""
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


def test_interp_default_silently_interpolates_garbage():
    out = _interp_frame().with_columns(
        polist.apply_interp("x", "y", pl.lit([0.5, 1.5])).alias("r")
    )
    assert out["r"][0].to_list() == [20.0, 20.0]


def test_interp_strict_raises_on_nan_x():
    df = pl.DataFrame({"x": [[0.0, NAN, 2.0]], "y": [[0.0, 1.0, 2.0]]})
    with pytest.raises(polars.exceptions.PolarsError, match="non-decreasing"):
        df.with_columns(
            polist.apply_interp("x", "y", pl.lit([0.5]), strict=True).alias("r")
        )


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


def test_agg_slices_default_inverted_range_is_empty_selection():
    df = pl.DataFrame({"v": [[1.0, 2.0]], "i": [[0.0, 1.0]]})
    out = df.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="mean", slices_include=[(2.0, 0.0)]
        ).alias("mean"),
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(2.0, 0.0)]
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


def test_agg_lists_default_truncates_silently():
    # list_length as a deliberate window: third elements are dropped.
    df = pl.DataFrame({"g": [1, 1], "v": [[1.0, 2.0, 30.0], [3.0, 4.0, 50.0]]})
    out = df.group_by("g").agg(
        polist.agg_lists("v", list_length=2, aggregation="mean").alias("r")
    )
    assert out["r"][0].to_list() == pytest.approx([2.0, 3.0])


def test_agg_lists_strict_passes_within_length():
    # Shorter lists stay legal under strict: they contribute nulls at
    # the missing positions, which is data, not a violation.
    df = pl.DataFrame({"g": [1, 1], "v": [[1.0, 2.0], [3.0]]})
    out = df.group_by("g").agg(
        polist.agg_lists("v", list_length=2, aggregation="mean", strict=True).alias("r")
    )
    assert out["r"][0].to_list() == pytest.approx([2.0, 2.0])
