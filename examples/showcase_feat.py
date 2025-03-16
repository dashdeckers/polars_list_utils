import numpy as np
import polars as pl
import polars_list_utils as polist


df = pl.DataFrame({
    'y_values': [
        [np.nan] + [1.0] + [0.0] * 10,
        [None] + [0.0] * 2,
        [np.nan] * 10,
    ]
})
print(df.with_columns(
    pl.col("y_values").list.len().alias("y_len"),
))

# shape: (3, 2)
# ┌───────────────────┬───────┐
# │ y_values          ┆ y_len │
# │ ---               ┆ ---   │
# │ list[f64]         ┆ u32   │
# ╞═══════════════════╪═══════╡
# │ [NaN, 1.0, … 0.0] ┆ 12    │
# │ [null, 0.0, 0.0]  ┆ 3     │
# │ [NaN, NaN, … NaN] ┆ 10    │
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
)
print(df)

# shape: (3, 4)
# ┌───────────────────┬───────────────────┬───────────────┬────────────────┐  
# │ y_values          ┆ x_axis            ┆ mean_of_range ┆ mean_of_offset │  
# │ ---               ┆ ---               ┆ ---           ┆ ---            │  
# │ list[f64]         ┆ list[f64]         ┆ f64           ┆ f64            │  
# ╞═══════════════════╪═══════════════════╪═══════════════╪════════════════╡  
# │ [NaN, 1.0, … 0.0] ┆ [0.0, 1.0, … 9.0] ┆ 1.0           ┆ 0.0            │  
# │ [null, 0.0, 0.0]  ┆ [0.0, 1.0, … 9.0] ┆ 0.0           ┆ 0.0            │  
# │ [NaN, NaN, … NaN] ┆ [0.0, 1.0, … 9.0] ┆ null          ┆ null           │  
# └───────────────────┴───────────────────┴───────────────┴────────────────┘