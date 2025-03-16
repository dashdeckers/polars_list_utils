import numpy as np
import polars as pl
import polars_list_utils as polist


df = pl.DataFrame({
    'list_col': [
        [None] + [0.0] * 1023,
        [np.nan] + [1.0] + [0.0] * 1022,
        [np.nan] * 1024,
    ]
})
print(df)

# shape: (3, 1)
# ┌────────────────────┐
# │ list_col           │
# │ ---                │
# │ list[f64]          │
# ╞════════════════════╡
# │ [null, 0.0, … 0.0] │
# │ [NaN, 1.0, … 0.0]  │
# │ [NaN, NaN, … NaN]  │
# └────────────────────┘

df = (
    df
    # Single group
    .group_by(pl.lit(1))
    .agg(
        # Elementwise mean
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=1024,
            aggregation="mean",
        ).alias('list_col_mean'),
        # Elementwise sum
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=1024,
            aggregation="sum",
        ).alias('list_col_sum'),
        # Elementwise count
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=1024,
            aggregation="count",
        ).alias('list_col_count'),
        # Elementwise mean but only for the first 2 elements of each list
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=2,
            aggregation="mean",
        ).alias('list_col_mean_shorter'),
    )
)
print(df)

# shape: (1, 5)
# ┌─────────┬────────────────┬───────────────┬───────────────┬───────────────┐
# │ literal ┆ list_col_mean  ┆ list_col_sum  ┆ list_col_coun ┆ list_col_mean │
# │ ---     ┆ ---            ┆ ---           ┆ t             ┆ _shorter      │
# │ i32     ┆ list[f64]      ┆ list[f64]     ┆ ---           ┆ ---           │
# │         ┆                ┆               ┆ list[f64]     ┆ list[f64]     │
# ╞═════════╪════════════════╪═══════════════╪═══════════════╪═══════════════╡
# │ 1       ┆ [null, 0.5, …  ┆ [null, 1.0, … ┆ [0.0, 2.0, …  ┆ [null, 0.5]   │
# │         ┆ 0.0]           ┆ 0.0]          ┆ 2.0]          ┆               │
# └─────────┴────────────────┴───────────────┴───────────────┴───────────────┘