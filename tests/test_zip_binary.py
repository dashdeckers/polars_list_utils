"""zip_binary: per element the result must be identical to the
corresponding scalar polars op -- the deference doctrine, executable."""
import math
import struct

import numpy as np
import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist

NAN = float("nan")
INF = float("inf")

# Adversarial element pairs: NaN on either and both sides, signed
# zeros, nulls, infinities, and division-by-zero fodder.
A = [1.0, 2.0, NAN, NAN, -0.0, 0.0, None, 1.0, None, INF, -INF, 5.0, 1.0, 0.0]
B = [2.0, 0.0, 5.0, NAN, 0.0, -0.0, 1.0, None, None, INF, INF, NAN, 0.0, 0.0]

FLOAT_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


def bits(v: float) -> bytes:
    """The IEEE bit pattern, so -0.0 != 0.0 and a NaN's sign is visible."""
    return struct.pack("<d", v)


def assert_row_matches(got: list, expected: list, ctx: str) -> None:
    # Element-wise ops are exact, so this compares bit patterns rather
    # than values: the adversarial pairs below include signed zeros and
    # NaNs of both signs, whose whole point a magnitude comparison
    # cannot see (-0.0 == 0.0, and every NaN equals every other).
    assert len(got) == len(expected), ctx
    for j, (g, e) in enumerate(zip(got, expected)):
        where = f"{ctx}, position {j}: got {g!r}, expected {e!r}"
        if e is None:
            assert g is None, where
        elif isinstance(e, float) and math.isnan(e):
            assert g is not None and math.isnan(g), where
        elif isinstance(e, bool):
            assert g == e, where
        else:
            assert g is not None and bits(g) == bits(e), where


@pytest.mark.parametrize("op", list(FLOAT_OPS))
def test_matches_polars_scalar_op(op):
    flat = pl.DataFrame(
        {"a": A, "b": B}, schema={"a": pl.Float64, "b": pl.Float64}
    )
    expected = flat.select(
        FLOAT_OPS[op](pl.col("a"), pl.col("b")).alias("r")
    )["r"].to_list()
    df = pl.DataFrame(
        {"a": [A], "b": [B]},
        schema={"a": pl.List(pl.Float64), "b": pl.List(pl.Float64)},
    )
    got = df.select(
        polist.zip_binary("a", "b", op=op).alias("r")
    )["r"][0].to_list()
    assert_row_matches(got, expected, op)


def test_kleene_and_or_match_polars():
    tri = [True, False, None]
    a = [x for x in tri for _ in tri]  # all 9 combinations
    b = tri * 3
    ops = {"and": lambda x, y: x & y, "or": lambda x, y: x | y}
    for op, expr_op in ops.items():
        flat = pl.DataFrame(
            {"a": a, "b": b}, schema={"a": pl.Boolean, "b": pl.Boolean}
        )
        expected = flat.select(
            expr_op(pl.col("a"), pl.col("b")).alias("r")
        )["r"].to_list()
        df = pl.DataFrame(
            {"a": [a], "b": [b]},
            schema={"a": pl.List(pl.Boolean), "b": pl.List(pl.Boolean)},
        )
        got = df.select(
            polist.zip_binary("a", "b", op=op).alias("r")  # ty: ignore[invalid-argument-type]
        )["r"][0].to_list()
        assert got == expected, op


@pytest.mark.parametrize("op", list(FLOAT_OPS))
def test_float32_values_match_polars_scalar_op(op):
    # The f32 path computes in f64 and rounds once at the exit cast,
    # while polars computes natively in f32. For +,-,*,/ double rounding
    # is provably innocuous, and this pins that the entry/exit casts do
    # not perturb values, NaNs or infinities.
    a32 = [float(np.float32(v)) if v is not None else None for v in A]
    b32 = [float(np.float32(v)) if v is not None else None for v in B]
    flat = pl.DataFrame(
        {"a": a32, "b": b32}, schema={"a": pl.Float32, "b": pl.Float32}
    )
    expected = flat.select(
        FLOAT_OPS[op](pl.col("a"), pl.col("b")).alias("r")
    )["r"].to_list()
    df = pl.DataFrame(
        {"a": [a32], "b": [b32]},
        schema={"a": pl.List(pl.Float32), "b": pl.List(pl.Float32)},
    )
    got = df.select(polist.zip_binary("a", "b", op=op).alias("r"))["r"][0].to_list()
    assert len(got) == len(expected), op
    for j, (g, e) in enumerate(zip(got, expected)):
        where = f"f32 {op}, position {j}: got {g!r}, expected {e!r}"
        if e is None:
            assert g is None, where
        elif isinstance(e, float) and math.isnan(e):
            assert g is not None and math.isnan(g), where
        elif isinstance(e, bool):
            assert g == e, where
        else:
            assert struct.pack("<f", g) == struct.pack("<f", e), where


def test_comparison_output_is_boolean():
    df = pl.DataFrame({"a": [[1.0]], "b": [[2.0]]})
    out = df.select(polist.zip_binary("a", "b", op="le").alias("r"))
    assert out["r"].dtype == pl.List(pl.Boolean)


def test_arithmetic_dtype_follows_supertyping():
    f32 = pl.DataFrame(
        {"a": [[1.0]], "b": [[2.0]]},
        schema={"a": pl.List(pl.Float32), "b": pl.List(pl.Float32)},
    )
    assert f32.select(polist.zip_binary("a", "b", op="mul").alias("r"))[
        "r"
    ].dtype == pl.List(pl.Float32)

    mixed = pl.DataFrame(
        {"a": [[1.0]], "b": [[2.0]]},
        schema={"a": pl.List(pl.Float32), "b": pl.List(pl.Float64)},
    )
    assert mixed.select(polist.zip_binary("a", "b", op="mul").alias("r"))[
        "r"
    ].dtype == pl.List(pl.Float64)


def test_integer_lists_raise():
    df = pl.DataFrame({"a": [[1, 2]], "b": [[3, 4]]})
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        df.select(polist.zip_binary("a", "b", op="add").alias("r"))


def test_arithmetic_on_booleans_raises():
    df = pl.DataFrame({"a": [[True]], "b": [[False]]})
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        df.select(polist.zip_binary("a", "b", op="add").alias("r"))


def test_and_on_floats_raises():
    df = pl.DataFrame({"a": [[1.0]], "b": [[0.0]]})
    with pytest.raises(polars.exceptions.PolarsError, match="Boolean"):
        df.select(polist.zip_binary("a", "b", op="and").alias("r"))


def test_dtype_errors_raise_at_plan_time():
    # Type-decidable violations must fail at schema resolution, before
    # any row is computed.
    lf = pl.LazyFrame({"a": [[1, 2]], "b": [[3, 4]]}).select(
        polist.zip_binary("a", "b", op="add").alias("r")
    )
    with pytest.raises(polars.exceptions.PolarsError, match="Float32 or Float64"):
        lf.collect_schema()


def test_length_mismatch_raises():
    df = pl.DataFrame({"a": [[1.0, 2.0]], "b": [[1.0]]})
    with pytest.raises(polars.exceptions.PolarsError, match="differ in length"):
        df.select(polist.zip_binary("a", "b", op="add").alias("r"))


def test_null_rows_stay_null():
    df = pl.DataFrame(
        {"a": [[1.0], None], "b": [None, [2.0]]},
        schema={"a": pl.List(pl.Float64), "b": pl.List(pl.Float64)},
    )
    out = df.select(polist.zip_binary("a", "b", op="add").alias("r"))
    assert out["r"].to_list() == [None, None]


def test_empty_lists_zip_to_empty():
    df = pl.DataFrame(
        {"a": [[]], "b": [[]]},
        schema={"a": pl.List(pl.Float64), "b": pl.List(pl.Float64)},
    )
    out = df.select(polist.zip_binary("a", "b", op="add").alias("r"))
    assert out["r"][0].to_list() == []


def test_literal_broadcasts_as_constant_template():
    df = pl.DataFrame({"a": [[1.0, 2.0], [3.0, 4.0]]})
    out = df.select(
        polist.zip_binary("a", pl.lit([10.0, 20.0]), op="add").alias("r")
    )
    assert out["r"].to_list() == [[11.0, 22.0], [13.0, 24.0]]


def test_left_side_literal_broadcasts():
    # The other side of the broadcasting axis; subtraction is not
    # commutative, so this also pins the operand order.
    df = pl.DataFrame({"b": [[1.0, 2.0], [3.0, 4.0]]})
    out = df.select(
        polist.zip_binary(pl.lit([10.0, 20.0]), "b", op="sub").alias("r")
    )
    assert out["r"].to_list() == [[9.0, 18.0], [7.0, 16.0]]


def test_broadcast_literal_length_mismatch_raises():
    df = pl.DataFrame({"a": [[1.0, 2.0], [3.0]]})
    with pytest.raises(polars.exceptions.PolarsError, match="differ in length"):
        df.select(polist.zip_binary("a", pl.lit([1.0, 2.0]), op="add"))


@pytest.mark.parametrize(
    ("op", "left", "right", "dtype"),
    [
        ("and", [[True], None], [None, [True]], pl.Boolean),
        ("or", [[True], None], [None, [True]], pl.Boolean),
        ("gt", [[1.0], None], [None, [1.0]], pl.Float64),
        ("eq", [[1.0], None], [None, [1.0]], pl.Float64),
    ],
)
def test_null_rows_stay_null_on_every_path(op, left, right, dtype):
    df = pl.DataFrame(
        {"a": left, "b": right},
        schema={"a": pl.List(dtype), "b": pl.List(dtype)},
    )
    out = df.select(polist.zip_binary("a", "b", op=op).alias("r"))
    assert out["r"].to_list() == [None, None]


@pytest.mark.parametrize(
    ("op", "dtype"),
    [("and", pl.Boolean), ("or", pl.Boolean), ("gt", pl.Float64), ("mul", pl.Float64)],
)
def test_empty_lists_zip_to_empty_on_every_path(op, dtype):
    df = pl.DataFrame(
        {"a": [[]], "b": [[]]},
        schema={"a": pl.List(dtype), "b": pl.List(dtype)},
    )
    out = df.select(polist.zip_binary("a", "b", op=op).alias("r"))
    assert out["r"][0].to_list() == []


def test_array_input_raises_cleanly():
    # PR 0 exists because an Array input used to abort the process; the
    # new plugins must also produce a catchable typed error.
    df = pl.DataFrame(
        {"a": [[1.0, 2.0]], "b": [[3.0, 4.0]]},
        schema={"a": pl.Array(pl.Float64, 2), "b": pl.Array(pl.Float64, 2)},
    )
    with pytest.raises(polars.exceptions.PolarsError, match="List column"):
        df.select(polist.zip_binary("a", "b", op="add"))


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_lazy_execution_matches_eager(engine):
    df = pl.DataFrame(
        {"a": [[float(i), float(i) + 1.0] for i in range(2000)]}
    )
    expr = polist.zip_binary("a", pl.lit([1.0, -1.0]), op="mul").alias("r")
    eager = df.select(expr)["r"].to_list()
    lazy = df.lazy().select(expr).collect(engine=engine)["r"].to_list()  # ty: ignore[not-subscriptable]
    assert lazy == eager
