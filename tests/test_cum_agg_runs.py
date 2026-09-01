"""cum_agg_runs: the scan oracle is the spec itself -- for each emitted
position, the vertical aggregation (computed by polars) over the run
prefix's non-null values -- run over an adversarial pattern matrix."""
import math

import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist

NAN = float("nan")

AGGS = ["sum", "mean", "median", "std", "min", "max", "delta", "count"]

VALUE_PATTERNS = [
    [],
    [1.0],
    [None],
    [NAN],
    [NAN, 1.0],
    [1.0, NAN, 0.5, 2.0],
    [1.0, None, 2.0],
    [None, None],
    [NAN, NAN],
    [0.0, -0.0],
    [1.0, -1.0, 0.0, None, NAN, 3.0, 2.5, -0.5],
]


def gate_patterns(n: int) -> list[list[bool | None]]:
    if n == 0:
        return [[]]
    pats: list[list[bool | None]] = [[True] * n, [False] * n, [None] * n]
    if n >= 2:
        pats += [
            ([True, False] * n)[:n],
            ([True, None] * n)[:n],
            [True] * (n // 2) + [False] * (n - n // 2),
            [None] + [True] * (n - 1),
            # A null gate between True elements must split the run.
            [True, None] + [True] * (n - 2),
        ]
    return pats


def vertical(agg: str, prefix: list[float]) -> float | int | None:
    """The vertical aggregation over a run prefix's non-null values,
    with polars itself as the kernel (the deference doctrine)."""
    s = pl.Series(prefix, dtype=pl.Float64)
    if agg == "sum":
        return s.sum()
    if agg == "count":
        return len(prefix)  # prefix holds only non-null values
    if agg == "delta":
        mx, mn = s.max(), s.min()
        return None if mx is None else mx - mn  # ty: ignore[unsupported-operator]
    return getattr(s, agg)()


def scan_oracle(agg, values, gates, outside) -> list:
    fill = 0 if outside == "zero" else None
    out: list = []
    prefix: list[float] = []
    prev_gate: object = object()  # unequal to True/False/None
    for v, g in zip(values, gates, strict=True):
        if g != prev_gate:
            prefix = []
        prev_gate = g
        if g is not True:
            out.append(fill)
        elif v is None:
            # Null values leave the state unchanged and emit null --
            # except count, which emits the unchanged running count.
            out.append(len(prefix) if agg == "count" else None)
        else:
            prefix.append(v)
            out.append(vertical(agg, prefix))
    return out


def assert_row_matches(got: list, expected: list, ctx: str) -> None:
    assert len(got) == len(expected), ctx
    for j, (g, e) in enumerate(zip(got, expected)):
        where = f"{ctx}, position {j}: got {g}, expected {e}"
        if e is None:
            assert g is None, where
        elif isinstance(e, float) and math.isnan(e):
            assert g is not None and math.isnan(g), where
        else:
            assert g == pytest.approx(e), where


@pytest.mark.parametrize("outside", ["null", "zero"])
@pytest.mark.parametrize("agg", AGGS)
def test_matches_scan_oracle(agg, outside):
    for values in VALUE_PATTERNS:
        for gates in gate_patterns(len(values)):
            df = pl.DataFrame(
                {"v": [values], "g": [gates]},
                schema={"v": pl.List(pl.Float64), "g": pl.List(pl.Boolean)},
            )
            got = df.select(
                polist.cum_agg_runs(
                    "v", "g", aggregation=agg, outside=outside
                ).alias("r")
            )["r"][0].to_list()
            expected = scan_oracle(agg, values, gates, outside)
            assert_row_matches(
                got, expected, f"agg={agg} outside={outside} v={values} g={gates}"
            )


@pytest.mark.parametrize(
    ("agg", "native"),
    [("sum", "cum_sum"), ("min", "cum_min"), ("max", "cum_max"), ("count", "cum_count")],
)
def test_all_true_gate_matches_native_cum(agg, native):
    # The §6.3 anchor, executable: with an all-True gate the scan
    # reduces to polars' native cumulative aggregation. (No leading-NaN
    # data here: polars leaks its fold seed there, see
    # test_leading_nan_follows_prefix_rule_not_polars_seed_leak.)
    data = [1.0, None, 2.0, NAN, 0.5, 2.0, None, -1.0]
    expected = getattr(pl.Series(data, dtype=pl.Float64), native)().to_list()
    df = pl.DataFrame({"v": [data]}, schema={"v": pl.List(pl.Float64)})
    got = df.select(
        polist.cum_agg_runs(
            "v", pl.lit([True] * len(data)), aggregation=agg
        ).alias("r")
    )["r"][0].to_list()
    assert_row_matches(got, expected, f"{agg} vs {native}")


def test_leading_nan_follows_prefix_rule_not_polars_seed_leak():
    # Native cum_max([NaN, 1.0]) emits -1.8e308 -- f64::MIN, its fold
    # seed -- at the NaN position: a sentinel leak, not any aggregation
    # of [NaN]. The prefix rule mandates the vertical max of [NaN],
    # which is NaN; this is a documented deviation from native cum_*.
    df = pl.DataFrame({"v": [[NAN, 1.0]], "g": [[True, True]]})
    for agg in ["min", "max", "delta"]:
        got = df.select(
            polist.cum_agg_runs("v", "g", aggregation=agg).alias("r")  # ty: ignore[invalid-argument-type]
        )["r"][0].to_list()
        assert math.isnan(got[0]), agg
    got = df.select(
        polist.cum_agg_runs("v", "g", aggregation="max").alias("r")
    )["r"][0].to_list()
    assert got[1] == 1.0


def test_runs_reset_after_false_gap():
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0, 4.0]], "g": [[True, True, False, True]]}
    )
    out = df.select(
        polist.cum_agg_runs("v", "g", aggregation="sum").alias("r")
    )["r"][0].to_list()
    assert out == [1.0, 3.0, None, 4.0]


def test_null_gate_breaks_runs():
    # True, null, True is three runs: the second True element starts fresh.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0]]},
        schema={"v": pl.List(pl.Float64)},
    ).with_columns(pl.lit([True, None, True]).alias("g"))
    out = df.select(
        polist.cum_agg_runs("v", "g", aggregation="sum").alias("r")
    )["r"][0].to_list()
    assert out == [1.0, None, 3.0]


def test_count_emits_running_count_at_null_values():
    # Matching native cum_count: [1.0, null, 2.0] -> [1, 1, 2].
    df = pl.DataFrame(
        {"v": [[1.0, None, 2.0]]}, schema={"v": pl.List(pl.Float64)}
    )
    out = df.select(
        polist.cum_agg_runs("v", pl.lit([True] * 3), aggregation="count").alias("r")
    )["r"][0].to_list()
    assert out == [1, 1, 2]


def test_outside_zero_fills_non_true_positions():
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0]]},
        schema={"v": pl.List(pl.Float64)},
    ).with_columns(pl.lit([True, False, None]).alias("g"))
    sums = df.select(
        polist.cum_agg_runs("v", "g", aggregation="sum", outside="zero").alias("r")
    )["r"][0].to_list()
    assert sums == [1.0, 0.0, 0.0]
    counts = df.select(
        polist.cum_agg_runs("v", "g", aggregation="count", outside="zero").alias("r")
    )["r"][0].to_list()
    assert counts == [1, 0, 0]


def test_std_needs_two_values_within_run():
    df = pl.DataFrame(
        {"v": [[1.0, 2.0, 3.0, 4.0]], "g": [[True, True, False, True]]}
    )
    out = df.select(
        polist.cum_agg_runs("v", "g", aggregation="std").alias("r")
    )["r"][0].to_list()
    assert out[0] is None  # one value
    assert out[1] == pytest.approx(pl.Series([1.0, 2.0]).std())
    assert out[2] is None  # outside
    assert out[3] is None  # new run, one value again


def test_float32_values_stay_float32():
    df = pl.DataFrame(
        {"v": [[1.0, 2.0]], "g": [[True, True]]},
        schema={"v": pl.List(pl.Float32), "g": pl.List(pl.Boolean)},
    )
    for agg in ["sum", "mean", "median", "std", "min", "max", "delta"]:
        out = df.select(
            polist.cum_agg_runs("v", "g", aggregation=agg).alias("r")  # ty: ignore[invalid-argument-type]
        )
        assert out["r"].dtype == pl.List(pl.Float32), agg


@pytest.mark.parametrize("inner", [pl.Float32, pl.Float64])
def test_count_emits_uint32(inner):
    df = pl.DataFrame(
        {"v": [[1.0, 2.0]], "g": [[True, True]]},
        schema={"v": pl.List(inner), "g": pl.List(pl.Boolean)},
    )
    out = df.select(polist.cum_agg_runs("v", "g", aggregation="count").alias("r"))
    assert out["r"].dtype == pl.List(pl.UInt32)
    assert out["r"][0].to_list() == [1, 2]


def test_integer_values_raise():
    df = pl.DataFrame({"v": [[1, 2]], "g": [[True, True]]})
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        df.select(polist.cum_agg_runs("v", "g", aggregation="sum").alias("r"))


def test_non_boolean_gate_raises():
    df = pl.DataFrame({"v": [[1.0, 2.0]], "g": [[1.0, 0.0]]})
    with pytest.raises(polars.exceptions.PolarsError, match="Boolean"):
        df.select(polist.cum_agg_runs("v", "g", aggregation="sum").alias("r"))


def test_dtype_errors_raise_at_plan_time():
    lf = pl.LazyFrame({"v": [[1.0]], "g": [[1.0]]}).select(
        polist.cum_agg_runs("v", "g", aggregation="sum").alias("r")
    )
    with pytest.raises(polars.exceptions.PolarsError, match="Boolean"):
        lf.collect_schema()


def test_length_mismatch_raises():
    df = pl.DataFrame({"v": [[1.0, 2.0]], "g": [[True]]})
    with pytest.raises(polars.exceptions.PolarsError, match="differ in length"):
        df.select(polist.cum_agg_runs("v", "g", aggregation="sum").alias("r"))


def test_null_rows_stay_null():
    df = pl.DataFrame(
        {"v": [[1.0], None], "g": [None, [True]]},
        schema={"v": pl.List(pl.Float64), "g": pl.List(pl.Boolean)},
    )
    out = df.select(polist.cum_agg_runs("v", "g", aggregation="sum").alias("r"))
    assert out["r"].to_list() == [None, None]


def test_empty_lists_scan_to_empty():
    df = pl.DataFrame(
        {"v": [[]], "g": [[]]},
        schema={"v": pl.List(pl.Float64), "g": pl.List(pl.Boolean)},
    )
    out = df.select(polist.cum_agg_runs("v", "g", aggregation="sum").alias("r"))
    assert out["r"][0].to_list() == []
    assert out["r"].dtype == pl.List(pl.Float64)


def test_gate_literal_broadcasts():
    df = pl.DataFrame({"v": [[1.0, 2.0], [3.0, 4.0]]})
    out = df.select(
        polist.cum_agg_runs("v", pl.lit([True, True]), aggregation="sum").alias("r")
    )
    assert out["r"].to_list() == [[1.0, 3.0], [3.0, 7.0]]


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_lazy_execution_matches_eager(engine):
    df = pl.DataFrame(
        {"v": [[float(i), float(i) + 1.0, float(i) + 2.0] for i in range(2000)]}
    )
    expr = polist.cum_agg_runs(
        "v", pl.lit([True, False, True]), aggregation="sum", outside="zero"
    ).alias("r")
    eager = df.select(expr)["r"].to_list()
    lazy = df.lazy().select(expr).collect(engine=engine)["r"].to_list()  # ty: ignore[not-subscriptable]
    assert lazy == eager
