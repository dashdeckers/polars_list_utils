"""A pipeline exercising every function must run on the streaming
engine with zero in-memory fallback nodes: a non-elementwise plugin
registration would silently materialize the whole stream at that node.

The streaming executor logs each physical node under POLARS_VERBOSE;
a logical subplan the engine cannot stream runs wrapped in an
"in-memory-map" node, which is the fallback marker this test forbids
("in-memory-source"/"in-memory-sink" are just the literal-frame source
and the result sink, and appear in every fully-streaming plan too).
"""
import os
import subprocess
import sys

PIPELINE = """
import polars as pl
import polars_list_utils as polist

Fs, N = 200.0, 64
FREQS = [i * Fs / N for i in range(N // 2 + 1)]
signal = [float(i % 7) for i in range(N)]

lf = (
    pl.LazyFrame({"grp": [1, 2], "signal": [signal, signal]})
    .with_columns(
        polist.apply_butterworth("signal", sample_rate=Fs, max_freq=50.0).alias("filt")
    )
    .with_columns(
        polist.apply_fft(
            "filt", sample_rate=Fs, window="hann", scaling="amplitude"
        ).alias("fft")
    )
    .with_columns(
        polist.apply_interp(
            pl.lit(FREQS), "fft", pl.lit([10.0, 20.0, 30.0])
        ).alias("interp")
    )
    .with_columns(
        polist.agg_slices(
            "fft",
            pl.lit(FREQS),
            aggregation="mean",
            slices_include=[(10.0, 30.0)],
            strict=True,
        ).alias("feat")
    )
    .with_columns(polist.zip_binary("fft", "fft", op="gt").alias("gates"))
    .with_columns(
        polist.cum_agg_runs(
            "fft", "gates", aggregation="sum", outside="zero"
        ).alias("cum")
    )
    .group_by("grp")
    .agg(
        polist.agg_lists(
            "interp", list_length=3, aggregation="mean", strict=True
        ).alias("agg")
    )
)
assert lf.collect(engine="streaming").height == 2
"""


def test_pipeline_streams_without_in_memory_fallback():
    env = os.environ.copy()
    env["POLARS_VERBOSE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", PIPELINE],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    log = proc.stderr + proc.stdout
    assert proc.returncode == 0, log
    assert "polars-stream" in log, f"streaming engine did not run:\n{log}"
    assert "in-memory-map" not in log, f"in-memory fallback node in plan:\n{log}"
