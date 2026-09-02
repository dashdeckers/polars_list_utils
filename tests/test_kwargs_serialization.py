"""Kwargs cross the plugin boundary as a pickle, and the decoder cannot
resolve a memoized *container* back-reference: any range object, bound
tuple, or range list that appears twice used to fail the query with
"recursive structure found". CPython's constant folding produces such
aliasing for perfectly ordinary literals, so the wrapper rebuilds every
range into fresh plain-float tuples. These are the shapes that broke."""
import numpy as np
import polars as pl
import pytest

import polars_list_utils as polist

NAN, INF = float("nan"), float("inf")


@pytest.fixture
def df_ramp() -> pl.DataFrame:
    ramp = [float(v) for v in range(11)]
    return pl.DataFrame({"v": [ramp], "i": [ramp]})


def test_identical_bound_tuples_within_one_range(df_ramp):
    # A single-point closed range: CPython folds the two identical
    # (1.0, "closed") constants into one object.
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="sum",
            slices_include=[((1.0, "closed"), (1.0, "closed"))],
        ).alias("r")
    )
    assert out["r"][0] == 1.0


def test_bound_tuple_shared_across_two_ranges(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="count",
            slices_include=[
                ((0.0, "closed"), (2.0, "open")),
                ((0.0, "closed"), (5.0, "open")),
            ],
        ).alias("r")
    )
    assert out["r"][0] == 5.0  # indices 0..4


def test_same_range_object_in_include_and_exclude(df_ramp):
    r = ((0.0, "closed"), (5.0, "open"))
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[r], slices_exclude=[r]
        ).alias("r")
    )
    assert out["r"][0] == 0.0


def test_same_list_object_for_both_kwargs(df_ramp):
    # Aliasing at the list level, not inside a range.
    ranges: list[polist.Range] = [((0.0, "closed"), (5.0, "open"))]
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="count",
            slices_include=ranges,
            slices_exclude=ranges,
        ).alias("r")
    )
    assert out["r"][0] == 0.0


def test_identical_bound_tuples_in_exclude_alone(df_ramp):
    # The rebuild must run on slices_exclude independently: a shared
    # object in BOTH kwargs stops aliasing once include is rebuilt, so
    # only an exclude-only case can catch a missing exclude rebuild.
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="count",
            slices_exclude=[((1.0, "closed"), (1.0, "closed"))],
        ).alias("r")
    )
    assert out["r"][0] == 10.0


def test_numpy_bounds_in_exclude_alone(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="count",
            slices_exclude=[(np.float64(2.0), np.float64(4.0))],
        ).alias("r")
    )
    assert out["r"][0] == 8.0


def test_bogus_boundary_mode_fails_with_the_parameter_name(df_ramp):
    # A typo mode must not surface serde untagged-enum internals.
    with pytest.raises(ValueError, match="boundary mode in slices_include"):
        polist.agg_slices(
            "v", "i",
            aggregation="count",
            slices_include=[((0.0, "clsoed"), (1.0, "open"))],  # ty: ignore[invalid-argument-type]
        )


def test_duplicate_simple_ranges(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(0.0, 2.0), (0.0, 2.0)]
        ).alias("r")
    )
    assert out["r"][0] == 3.0


def test_numpy_scalar_bounds(df_ramp):
    # np.float64 pickles as a reduce the decoder cannot resolve, so the
    # wrapper must coerce bounds to plain floats.
    lo, hi = np.float64(2.0), np.float64(4.0)
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=[(lo, hi)]
        ).alias("r")
    )
    assert out["r"][0] == 9.0  # 2 + 3 + 4


def test_integer_bounds(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="sum", slices_include=[(2, 4)]
        ).alias("r")
    )
    assert out["r"][0] == 9.0


def test_list_shaped_ranges(df_ramp):
    # The shape you get from json.load.
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i",
            aggregation="sum",
            slices_include=[[[2.0, "closed"], [4.0, "closed"]]],  # ty: ignore[invalid-argument-type]
        ).alias("r")
    )
    assert out["r"][0] == 9.0


def test_infinite_and_nan_bounds_still_work(df_ramp):
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(-INF, INF)]
        ).alias("all"),
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=[(NAN, NAN)], strict=False
        ).alias("none"),
    )
    assert out["all"][0] == 11.0
    assert out["none"][0] == 0.0


def test_many_ranges(df_ramp):
    ranges: list[polist.Range] = [
        ((float(i), "closed"), (float(i), "closed")) for i in range(300)
    ]
    out = df_ramp.with_columns(
        polist.agg_slices(
            "v", "i", aggregation="count", slices_include=ranges
        ).alias("r")
    )
    assert out["r"][0] == 11.0
