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

df = (
    df
    .group_by(pl.lit(1))
    .agg(
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=1024,
            aggregation="mean",
        ).alias('list_col_mean'),
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=1024,
            aggregation="sum",
        ).alias('list_col_sum'),
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=1024,
            aggregation="count",
        ).alias('list_col_count'),
        polist.aggregate_list_col_elementwise(
            'list_col',
            list_size=2,
            aggregation="mean",
        ).alias('list_col_mean_shorter'),
    )
)
print(df)

