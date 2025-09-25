import numpy as np
import polars as pl
import polars_list_utils as polist


df = pl.DataFrame({
    'y_values': [
        [np.nan] + [1.0] + [0.0] * 10,
        [None] + [0.0] * 2,
        [np.nan] * 10,
        [1, 8, 3, 2],
        [4, 5, 2, 3],
    ]
})
print(df.with_columns(
    pl.col("y_values").list.len().alias("y_len"),
))

# shape: (5, 2)
# ┌───────────────────┬───────┐
# │ y_values          ┆ y_len │
# │ ---               ┆ ---   │
# │ list[f64]         ┆ u32   │
# ╞═══════════════════╪═══════╡
# │ [NaN, 1.0, … 0.0] ┆ 12    │
# │ [null, 0.0, 0.0]  ┆ 3     │
# │ [NaN, NaN, … NaN] ┆ 10    │
# │ [1.0, 8.0, … 2.0] ┆ 4     │
# │ [4.0, 5.0, … 3.0] ┆ 4     │
# └───────────────────┴───────┘

df = (
    df
    # This will be our x_axis for the mean_of_range (simple indices here)
    .with_columns(
        pl.lit(list(np.arange(10))).cast(pl.List(pl.Float64))
        .alias('x_axis'),
    )
    .with_columns(
        polist.mean_of_range(
            list_column_y='y_values',
            list_column_x='x_axis',
            # Take y_values where x_axis is between 0 and 1 (inclusive)
            x_min=0,
            x_max=1,
        ).alias('mean_of_range'),
    )
    .with_columns(
        polist.mean_of_range(
            list_column_y='y_values',
            list_column_x='x_axis',
            # Take y_values where x_axis is between 0 and 3 (inclusive)
            x_min=0,
            x_max=3,
            # Skip the first index
            x_min_idx_offset=1,
        ).alias('mean_of_offset'),
    )
    .with_columns(
        polist.agg_of_range(
            list_column_y='y_values',
            list_column_x='x_axis',
            aggregation="median",
            # Take y_values where x_axis is between 0 and 3 (inclusive)
            x_min=0,
            x_max=3,
        ).alias('median'),
    )
    .with_columns(
        polist.agg_of_range(
            list_column_y='y_values',
            list_column_x='x_axis',
            aggregation="median",
            x_range_excluded=(1, 2),
            # Take y_values where x_axis is between 0 and 3 (inclusive)
            x_min=0,
            x_max=3,
        ).alias('median_exclude_inner'),
    )
)
print(df)

# shape: (5, 6)
# ┌───────────────────┬───────────────────┬───────────────┬────────────────┬────────┬──────────────────────┐
# │ y_values          ┆ x_axis            ┆ mean_of_range ┆ mean_of_offset ┆ median ┆ median_exclude_inner │
# │ ---               ┆ ---               ┆ ---           ┆ ---            ┆ ---    ┆ ---                  │
# │ list[f64]         ┆ list[f64]         ┆ f64           ┆ f64            ┆ f64    ┆ f64                  │
# ╞═══════════════════╪═══════════════════╪═══════════════╪════════════════╪════════╪══════════════════════╡
# │ [NaN, 1.0, … 0.0] ┆ [0.0, 1.0, … 9.0] ┆ 1.0           ┆ 0.0            ┆ 0.0    ┆ 0.0                  │
# │ [null, 0.0, 0.0]  ┆ [0.0, 1.0, … 9.0] ┆ 0.0           ┆ 0.0            ┆ 0.0    ┆ null                 │
# │ [NaN, NaN, … NaN] ┆ [0.0, 1.0, … 9.0] ┆ null          ┆ null           ┆ null   ┆ null                 │
# │ [1.0, 8.0, … 2.0] ┆ [0.0, 1.0, … 9.0] ┆ 4.5           ┆ 4.333333       ┆ 2.5    ┆ 1.5                  │
# │ [4.0, 5.0, … 3.0] ┆ [0.0, 1.0, … 9.0] ┆ 4.5           ┆ 3.333333       ┆ 3.5    ┆ 3.5                  │
# └───────────────────┴───────────────────┴───────────────┴────────────────┴────────┴──────────────────────┘