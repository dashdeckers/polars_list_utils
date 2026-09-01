"""cum_agg_runs: the scan oracle is the spec itself -- for each emitted
position, the vertical aggregation (computed by polars) over the run
prefix's non-null values -- run over an adversarial pattern matrix."""
import math

import numpy as np
import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist

NAN = float("nan")

AGGS = ["sum", "mean", "median", "std", "min", "max", "delta", "count"]

INF = float("inf")

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
    [-0.0, 0.0],
    [-0.0, -0.0],
    [1.0, -1.0, 0.0, None, NAN, 3.0, 2.5, -0.5],
    # Infinities and overflow magnitudes: legitimate floats per spec
    # 6.1, and the regime where a naive midpoint or an incremental
    # variance parts company with the vertical kernels.
    [INF, INF],
    [-INF, -INF],
    [1.0, -INF],
    [-INF, 3.0],
    [-INF, INF],
    [1e308, 1e308],
    [-1e308, 1e308],
    [1e308, 1e308, 1.0, 2.0],
    [1e16, 1e16 + 2.0],
    [INF, 1.0, NAN, -INF, None, 2.0],
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


def two_pass_std(prefix: list[float]) -> float | None:
    """Sample std (ddof=1) the way the library's vertical kernel computes
    it: mean first, then the squared deviations.

    The oracle cannot use polars here. polars' own std is neither
    two-pass nor Welford nor naive — it disagrees with all three once
    its intermediate accumulation overflows (`[1e308, 1e308]` reads NaN)
    — and no algorithm reproduces it bit-for-bit anyway, because it
    accumulates in lanes. The library's contract is therefore that the
    scan equals the *library's* vertical kernel exactly (asserted here),
    which in turn tracks polars on ordinary data (asserted by
    test_std_tracks_polars_on_ordinary_data).
    """
    n = len(prefix)
    if n < 2:
        return None
    mean = sum(prefix) / n
    # d * d, not d ** 2: Python's power operator raises OverflowError
    # where IEEE multiplication saturates to inf, as the kernel does.
    variance = sum((v - mean) * (v - mean) for v in prefix) / (n - 1)
    return math.sqrt(variance) if variance == variance and variance >= 0 else NAN


def vertical(agg: str, prefix: list[float]) -> float | int | None:
    """The vertical aggregation over a run prefix's non-null values,
    with polars itself as the kernel (the deference doctrine)."""
    s = pl.Series(prefix, dtype=pl.Float64)
    if agg == "sum":
        return s.sum()
    if agg == "count":
        return len(prefix)  # prefix holds only non-null values
    if agg == "std":
        return two_pass_std(prefix)
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
        where = f"{ctx}, position {j}: got {g!r}, expected {e!r}"
        if e is None:
            assert g is None, where
        elif isinstance(e, float) and math.isnan(e):
            assert g is not None and math.isnan(g), where
        elif e == 0.0:
            # Pin the sign of a zero too: min/max keep the first of a
            # signed-zero tie, and sum folds from +0.0, both of which a
            # magnitude-only comparison cannot see.
            assert g == 0.0 and math.copysign(1.0, g) == math.copysign(1.0, e), where
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


def test_std_tracks_polars_on_ordinary_data():
    # The other half of the std contract: the library's kernel agrees
    # with polars' vertical std to floating-point rounding whenever
    # polars' own accumulation stays in range.
    rng = np.random.default_rng(3)
    data = list(rng.standard_normal(60) * 1000 + 1e6)
    got = pl.DataFrame({"v": [data]}, schema={"v": pl.List(pl.Float64)}).select(
        polist.cum_agg_runs(
            "v", pl.lit([True] * len(data)), aggregation="std"
        ).alias("r")
    )["r"][0].to_list()
    for k in range(2, len(data) + 1):
        assert got[k - 1] == pytest.approx(
            pl.Series(data[:k], dtype=pl.Float64).std(), rel=1e-12
        ), k


def test_std_is_the_same_kernel_as_agg_slices():
    # One input, one answer: the scan's final position must equal the
    # library's own vertical aggregation over the whole run, including
    # in the overflow regime where polars' std parts company with every
    # textbook algorithm.
    for values in [[1e308, 1e308], [1e16, 1e16 + 2.0], [1.0, 2.0, 3.0], [1e-8, 2e-8]]:
        idx = [float(i) for i in range(len(values))]
        df = pl.DataFrame(
            {"v": [values], "i": [idx]},
            schema={"v": pl.List(pl.Float64), "i": pl.List(pl.Float64)},
        )
        scan = df.select(
            polist.cum_agg_runs(
                "v", pl.lit([True] * len(values)), aggregation="std"
            ).alias("r")
        )["r"][0].to_list()[-1]
        flat = df.select(
            polist.agg_slices("v", "i", aggregation="std").alias("r")
        )["r"][0]
        assert scan == flat, values


def test_std_does_not_collapse_to_zero_for_distinct_values():
    # An incremental variance whose running mean rounds onto the next
    # value reports zero spread for data that has spread; at timestamp
    # magnitudes that is an ordinary input, not an exotic one.
    values = [1e16, 1e16 + 2.0]
    got = pl.DataFrame({"v": [values]}, schema={"v": pl.List(pl.Float64)}).select(
        polist.cum_agg_runs("v", pl.lit([True, True]), aggregation="std").alias("r")
    )["r"][0].to_list()
    assert got[1] == pytest.approx(pl.Series(values, dtype=pl.Float64).std())
    assert got[1] > 0.0


def test_median_matches_polars_at_extremes():
    # A naive (a + b) / 2 midpoint overflows for large equal values and
    # loses the equality short-circuit that equal infinities need.
    for values in [
        [1e308, 1e308], [-1e308, -1e308], [1.0, -INF], [-INF, 3.0],
        [INF, INF], [-INF, -INF], [-1e308, 1e308], [1.0, INF],
    ]:
        got = pl.DataFrame({"v": [values]}, schema={"v": pl.List(pl.Float64)}).select(
            polist.cum_agg_runs(
                "v", pl.lit([True] * len(values)), aggregation="median"
            ).alias("r")
        )["r"][0].to_list()
        expected = [
            pl.Series(values[:k], dtype=pl.Float64).median()
            for k in range(1, len(values) + 1)
        ]
        assert_row_matches(got, expected, f"median {values}")


@pytest.mark.parametrize("values", [[-0.0, 0.0], [0.0, -0.0], [-0.0, -0.0]])
def test_signed_zero_ties_keep_the_first_value(values):
    # polars' min/max keep the earlier of two values that compare equal;
    # Rust's f64::min/max keep the later, which flips the sign of zero.
    df = pl.DataFrame({"v": [values]}, schema={"v": pl.List(pl.Float64)})
    for agg in ["min", "max", "median"]:
        got = df.select(
            polist.cum_agg_runs(
                "v", pl.lit([True] * len(values)), aggregation=agg  # ty: ignore[invalid-argument-type]
            ).alias("r")
        )["r"][0].to_list()
        expected = [
            getattr(pl.Series(values[:k], dtype=pl.Float64), agg)()
            for k in range(1, len(values) + 1)
        ]
        assert_row_matches(got, expected, f"{agg} {values}")


def test_leading_infinity_also_deviates_from_native_cum_seed_leak():
    # The native seed leak is not NaN-specific: cum_min/cum_max fold
    # from -+f64::MAX and never replace the seed on a tie, so a leading
    # infinity of the same sign leaks it too. The prefix rule gives the
    # true infinity instead.
    df = pl.DataFrame({"v": [[INF, 1.0]]}, schema={"v": pl.List(pl.Float64)})
    got = df.select(
        polist.cum_agg_runs("v", pl.lit([True, True]), aggregation="min").alias("r")
    )["r"][0].to_list()
    native = pl.Series([INF, 1.0], dtype=pl.Float64).cum_min().to_list()
    assert got[0] == INF
    assert native[0] == pytest.approx(1.7976931348623157e308)


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


def test_value_literal_broadcasts():
    # The other side of the broadcasting axis: one constant signal
    # scanned against per-row gates.
    df = pl.DataFrame(
        {"g": [[True, True], [True, False]]}, schema={"g": pl.List(pl.Boolean)}
    )
    out = df.select(
        polist.cum_agg_runs(pl.lit([1.0, 2.0]), "g", aggregation="sum").alias("r")
    )
    assert out["r"].to_list() == [[1.0, 3.0], [1.0, None]]


def test_run_state_resets_after_a_nan_poisoned_run():
    # A NaN poisons sum/mean/std for the rest of its run; a gate break
    # must start the next run clean rather than carrying it over.
    df = pl.DataFrame(
        {"v": [[1.0, NAN, 2.0, 5.0, 7.0]]}, schema={"v": pl.List(pl.Float64)}
    ).with_columns(pl.lit([True, True, True, False, True]).alias("g"))
    sums = df.select(
        polist.cum_agg_runs("v", "g", aggregation="sum").alias("r")
    )["r"][0].to_list()
    assert sums[0] == 1.0
    assert math.isnan(sums[1]) and math.isnan(sums[2])  # poisoned
    assert sums[3] is None  # outside
    assert sums[4] == 7.0  # fresh run, uncontaminated

    means = df.select(
        polist.cum_agg_runs("v", "g", aggregation="mean").alias("r")
    )["r"][0].to_list()
    assert math.isnan(means[2]) and means[4] == 7.0


def test_array_input_is_supported():
    # Array used to abort the process, then raised; it is now supported.
    # A mixed Array value column and List gate is allowed, and the output
    # follows the value column. See tests/test_array_container.py.
    df = pl.DataFrame(
        {"v": [[1.0, 2.0]], "g": [[True, True]]},
        schema={"v": pl.Array(pl.Float64, 2), "g": pl.List(pl.Boolean)},
    )
    out = df.select(polist.cum_agg_runs("v", "g", aggregation="sum").alias("r"))
    assert out["r"].dtype == pl.Array(pl.Float64, 2)
    assert out["r"][0].to_list() == [1.0, 3.0]


def test_float32_accumulates_in_f64_and_rounds_once():
    # Doctrine 4: the kernel computes in f64 and rounds at the exit
    # boundary, while polars' Float32 cum_sum rounds every step. The
    # deviation is deliberate, so pin it rather than let it drift.
    # 3e38 + 3e38 overflows Float32 (max ~3.4e38) but not Float64.
    values = [3e38, 3e38, -3e38]
    df = pl.DataFrame({"v": [values]}, schema={"v": pl.List(pl.Float32)})
    got = df.select(
        polist.cum_agg_runs("v", pl.lit([True] * 3), aggregation="sum").alias("r")
    )["r"][0].to_list()
    native = pl.Series(values, dtype=pl.Float32).cum_sum().to_list()
    # polars overflows at the second step and never recovers; the f64
    # accumulator stays in range and rounds back to a finite Float32.
    assert math.isinf(native[1]) and math.isinf(native[2])
    assert math.isinf(got[1]) and got[2] == pytest.approx(3e38, rel=1e-6)


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
