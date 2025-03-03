use polars::{prelude::*, series::amortized_iter::AmortSeries};
use pyo3_polars::export::polars_core::utils::align_chunks_binary;

pub fn same_dtype(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

/// From the [plugin tutorial](https://marcogorelli.github.io/polars-plugins-tutorial/lists/)
/// by Marco Gorelli:
/// Polars Series are backed by chunked arrays.
/// align_chunks_binary just ensures that the chunks have the same lengths. It may need to rechunk
/// under the hood for us; amortized_iter returns an iterator of AmortSeries, each of which
/// corresponds to a row from our input.
pub fn binary_amortized_elementwise<'a, T, K, F>(
    lhs: &'a ListChunked,
    rhs: &'a ListChunked,
    mut f: F,
) -> ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: ArrayFromIter<Option<K>>,
    F: FnMut(&AmortSeries, &AmortSeries) -> Option<K> + Copy,
{
    {
        let (lhs, rhs) = align_chunks_binary(lhs, rhs);
        lhs.amortized_iter()
            .zip(rhs.amortized_iter())
            .map(|(lhs, rhs)| match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => f(&lhs, &rhs),
                _ => None,
            })
            .collect_ca(PlSmallStr::EMPTY)
    }
}
