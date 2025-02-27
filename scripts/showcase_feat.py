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
    .with_columns(
        pl.lit(list(np.arange(1024))).cast(pl.List(pl.Float64))
        .alias('list_col_x'),
    )
    .with_columns(
        polist.mean_of_range(
            list_column_y='list_col',
            list_column_x='list_col_x',
            x_min=0,
            x_max=2,
        ).alias('list_col_mean_of_range'),
    )
)
print(df)

