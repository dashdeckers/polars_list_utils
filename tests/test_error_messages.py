"""Plugins are handed one chunk at a time, so a position counted inside
the kernel is chunk-local. Naming it as an absolute row number sends the
reader to a different, perfectly valid row -- and which row it names
changes with chunk layout and with the engine. These messages therefore
describe the offending row by its shape and never by its position."""
import polars as pl
import polars.exceptions
import pytest

import polars_list_utils as polist


def message(fn) -> str:
    with pytest.raises(polars.exceptions.PolarsError) as excinfo:
        fn()
    return str(excinfo.value)


def multi_chunk_frame(offending_at: int, n: int = 1000, chunks: int = 4):
    """A frame whose only malformed row sits in the last chunk."""
    parts = []
    for c in range(chunks):
        rows = [[1.0, 2.0] for _ in range(n)]
        gates = [[True, True] for _ in range(n)]
        if c == chunks - 1:
            rows[offending_at - c * n] = [1.0, 2.0, 3.0]
        parts.append(
            pl.DataFrame(
                {"v": rows, "g": gates},
                schema={"v": pl.List(pl.Float64), "g": pl.List(pl.Boolean)},
            )
        )
    return pl.concat(parts, rechunk=False)


def test_cum_agg_runs_length_mismatch_names_no_row():
    df = multi_chunk_frame(offending_at=3500)
    assert df.n_chunks() > 1
    msg = message(lambda: df.select(polist.cum_agg_runs("v", "g", aggregation="sum")))
    assert "differ in length (3 vs 2)" in msg
    assert "at row" not in msg


def test_zip_binary_length_mismatch_names_no_row():
    df = pl.DataFrame(
        {"a": [[1.0, 2.0], [3.0]], "b": [[1.0, 2.0], [1.0, 2.0]]},
        schema={"a": pl.List(pl.Float64), "b": pl.List(pl.Float64)},
    )
    msg = message(lambda: df.select(polist.zip_binary("a", "b", op="add")))
    assert "differ in length (1 vs 2)" in msg
    assert "at row" not in msg


def test_message_is_stable_across_chunk_layout_and_engine():
    # The regression this guards: the reported text used to change with
    # chunking and with the engine, for identical data.
    df = multi_chunk_frame(offending_at=3500)
    chunked = message(lambda: df.select(polist.cum_agg_runs("v", "g", aggregation="sum")))
    rechunked = message(
        lambda: df.rechunk().select(polist.cum_agg_runs("v", "g", aggregation="sum"))
    )
    streamed = message(
        lambda: df.lazy()
        .select(polist.cum_agg_runs("v", "g", aggregation="sum"))
        .collect(engine="streaming")
    )
    assert chunked == rechunked
    assert "differ in length (3 vs 2)" in streamed


def test_agg_lists_strict_message_names_no_row():
    df = pl.DataFrame({"g": [1, 1], "v": [[1.0, 2.0], [1.0, 2.0, 3.0]]})
    msg = message(
        lambda: df.group_by("g").agg(
            polist.agg_lists("v", list_length=2, aggregation="mean", strict=True)
        )
    )
    assert "list length 3 but list_length=2" in msg
    assert "row 1" not in msg


def test_apply_interp_strict_message_names_no_row():
    df = pl.DataFrame(
        {"x": [[0.0, 1.0], [2.0, 1.0]], "y": [[0.0, 1.0], [0.0, 1.0]]}
    )
    msg = message(
        lambda: df.select(
            polist.apply_interp("x", "y", pl.lit([0.5]), strict=True)
        )
    )
    assert "found 2 followed by 1" in msg
    assert "at row" not in msg
