use polars::prelude::*;

/// Output dtype for list-returning expressions.
pub(crate) fn list_f64_output(_: &[Field]) -> PolarsResult<Field> {
    Ok(Field::new(
        PlSmallStr::from(""),
        DataType::List(Box::new(DataType::Float64)),
    ))
}

/// Extract one row's list as a `Vec<f64>`.
///
/// Returns `None` (-> null output row) if the list is empty or contains
/// null elements: dropping inner nulls would silently misalign this list
/// against its paired columns.
fn extract_row(s: &Series) -> Option<Vec<f64>> {
    let values: Option<Vec<f64>> = s.f64().ok()?.into_iter().collect();
    values.filter(|v| !v.is_empty())
}

/// Extract one row's list with inner nulls preserved.
///
/// Returns `None` (-> null output row) only if the list is empty.
fn extract_row_nullable(s: &Series) -> Option<Vec<Option<f64>>> {
    let values: Vec<Option<f64>> = s.f64().ok()?.into_iter().collect();
    (!values.is_empty()).then_some(values)
}

/// Pre-extracted list-column inputs with automatic literal broadcasting.
///
/// Polars expression plugins receive `&[Series]` as-is from the engine,
/// without broadcasting length-1 literals. This struct handles that
/// uniformly: all inputs must be `List` columns with a numeric inner
/// dtype (cast to f64), of either length N or length 1 (broadcast to N).
///
/// [`ListInputs::new`] nulls out rows whose lists contain null elements
/// (for transforms, where a missing sample invalidates the signal);
/// [`ListInputs::with_inner_nulls`] preserves them as `Option<f64>`
/// (for aggregations, which skip missing data polars-style).
pub(crate) struct ListInputs<T> {
    columns: Vec<Vec<Option<Vec<T>>>>,
    len: usize,
}

impl ListInputs<f64> {
    pub(crate) fn new(inputs: &[Series]) -> PolarsResult<Self> {
        Self::build(inputs, extract_row)
    }
}

impl ListInputs<Option<f64>> {
    pub(crate) fn with_inner_nulls(inputs: &[Series]) -> PolarsResult<Self> {
        Self::build(inputs, extract_row_nullable)
    }
}

impl<T> ListInputs<T> {
    fn build(
        inputs: &[Series],
        extract: impl Fn(&Series) -> Option<Vec<T>>,
    ) -> PolarsResult<Self> {
        polars_ensure!(
            !inputs.is_empty(),
            ComputeError: "expected at least one input"
        );

        let list_f64 = DataType::List(Box::new(DataType::Float64));
        let cas = inputs
            .iter()
            .map(|s| {
                polars_ensure!(
                    matches!(s.dtype(), DataType::List(inner) if inner.is_primitive_numeric()),
                    ComputeError: "expected List column with numeric inner dtype, got {}",
                    s.dtype()
                );
                s.cast(&list_f64)
            })
            .collect::<PolarsResult<Vec<Series>>>()?;

        // The broadcast length: length-1 inputs are literals, all others must agree.
        let len = cas
            .iter()
            .map(|s| s.len())
            .filter(|&l| l != 1)
            .max()
            .unwrap_or(1);
        for s in &cas {
            polars_ensure!(
                s.len() == len || s.len() == 1,
                ComputeError: "all inputs must have length {len} or 1 (got {})",
                s.len()
            );
        }

        let columns = cas
            .iter()
            .map(|s| {
                Ok(s.list()?
                    .into_iter()
                    .map(|opt_s| opt_s.and_then(|s| extract(&s)))
                    .collect())
            })
            .collect::<PolarsResult<_>>()?;

        Ok(Self { columns, len })
    }

    /// Number of output rows (the broadcast length).
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// All column values for row `i`, with length-1 inputs broadcast.
    /// Returns `None` if any column is null (or empty) at this row.
    pub(crate) fn row(
        &self,
        i: usize,
    ) -> Option<Vec<&[T]>> {
        self.columns
            .iter()
            .map(|col| col[if col.len() == 1 { 0 } else { i }].as_deref())
            .collect()
    }
}

/// Apply a fallible transformation elementwise across one or more list
/// columns, returning a `List[f64]` column.
///
/// The closure receives one `&[f64]` per input column (length-1 literals
/// broadcast) and returns `Ok(None)` to emit a null row, or `Err` to
/// abort the whole query. Null input rows stay null.
pub(crate) fn apply_list_transform<F>(
    inputs: &[Series],
    f: F,
) -> PolarsResult<Series>
where
    F: Fn(&[&[f64]]) -> PolarsResult<Option<Vec<f64>>>,
{
    let li = ListInputs::new(inputs)?;

    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        PlSmallStr::from(""),
        li.len(),
        li.len() * 8,
        DataType::Float64,
    );

    for i in 0..li.len() {
        match li.row(i).map(|slices| f(&slices)).transpose()?.flatten() {
            Some(result) => builder.append_slice(&result),
            None => builder.append_null(),
        }
    }

    Ok(builder.finish().into_series())
}
