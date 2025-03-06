from pathlib import Path
from typing import Optional, Union, Literal

import polars as pl
from polars.plugins import register_plugin_function

from polars_list_utils._internal import __version__ as __version__


root_path = Path(__file__).parent


def apply_fft(
    list_column: Union[pl.Expr, str, pl.Series],
    sample_rate: int,
    window: Optional[str] = None,
    bp_min: Optional[float] = None,
    bp_max: Optional[float] = None,
    bp_ord: Optional[int] = None,
    skip_fft: bool = False,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "sample_rate": sample_rate,
            "window": window,
            "bp_min": bp_min,
            "bp_max": bp_max,
            "bp_ord": bp_ord,
            "skip_fft": skip_fft,
        },
        plugin_path=root_path,
        function_name="expr_fft",
        is_elementwise=True,
    )


def get_freqs(
    list_column: Union[pl.Expr, str, pl.Series],
    sample_rate: int,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "sample_rate": sample_rate,
        },
        plugin_path=root_path,
        function_name="expr_fft_freqs",
        is_elementwise=True,
    )


def normalize_fft(
    list_column: Union[pl.Expr, str, pl.Series],
    norm_column: Union[pl.Expr, str, pl.Series],
    max_norm_val: float,
    sample_rate: int,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column, norm_column],
        kwargs={
            "max_norm_val": max_norm_val,
            "sample_rate": sample_rate,
        },
        plugin_path=root_path,
        function_name="expr_normalize_ffts",
        is_elementwise=True,
    )


def get_normalized_freqs(
    list_column: Union[pl.Expr, str, pl.Series],
    max_norm_val: float,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "max_norm_val": max_norm_val,
        },
        plugin_path=root_path,
        function_name="expr_fft_normalized_freqs",
        is_elementwise=True,
    )


def aggregate_list_col_elementwise(
    list_column: Union[pl.Expr, str, pl.Series],
    list_size: int,
    aggregation: Literal["mean", "sum", "count"] = "mean",
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column],
        kwargs={
            "list_size": list_size,
            "aggregation": aggregation,
        },
        plugin_path=root_path,
        function_name="expr_aggregate_list_col_elementwise",
        is_elementwise=False,
        returns_scalar=True,
    )


def mean_of_range(
    list_column_y: Union[pl.Expr, str, pl.Series],
    list_column_x: Union[pl.Expr, str, pl.Series],
    x_min: float,
    x_max: float,
    x_min_idx_offset: Optional[int] = None,
    x_max_idx_offset: Optional[int] = None,
) -> pl.Expr:
    return register_plugin_function(
        args=[list_column_y, list_column_x],
        kwargs={
            "x_min": x_min,
            "x_max": x_max,
            "x_min_idx_offset": x_min_idx_offset,
            "x_max_idx_offset": x_max_idx_offset,
        },
        plugin_path=root_path,
        function_name="expr_mean_of_range",
        is_elementwise=True,
    )