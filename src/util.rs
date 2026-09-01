use polars::prelude::*;
use std::cmp::Ordering;

/// Total-order float comparison matching polars: NaN compares equal to
/// NaN and greater than everything else (regardless of NaN sign bit),
/// while `-0.0 == 0.0` as in IEEE — both unlike `f64::total_cmp`.
pub(crate) fn tot_cmp(
    a: f64,
    b: f64,
) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).expect("non-NaN floats compare"),
    }
}

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

/// The broadcast length across inputs: length-1 inputs are literals,
/// all others must agree.
pub(crate) fn broadcast_len(lens: &[usize]) -> PolarsResult<usize> {
    let len = lens.iter().copied().filter(|&l| l != 1).max().unwrap_or(1);
    for &l in lens {
        polars_ensure!(
            l == len || l == 1,
            ComputeError: "all inputs must have length {len} or 1 (got {l})"
        );
    }
    Ok(len)
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

        let lens = cas.iter().map(|s| s.len()).collect::<Vec<_>>();
        let len = broadcast_len(&lens)?;

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
/// The closure receives one `&[f64]` per input column (length-1
/// literals broadcast) and returns `Ok(None)` to emit a null row, or
/// `Err` to abort the whole query. Null input rows stay null.
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

/// One `List` input for the dtype-preserving entry layer used by
/// `zip_binary` and `cum_agg_runs`.
///
/// Unlike [`ListInputs`], rows keep inner nulls **and** empty lists
/// (these functions are total over zero elements: empty in, empty out),
/// and the validated original inner dtype is kept so the kernel's `f64`
/// result can be restored to it at the exit boundary.
pub(crate) struct TypedListInput<T> {
    rows: Vec<Option<Vec<Option<T>>>>,
    inner_dtype: DataType,
}

impl TypedListInput<f64> {
    /// Extract a float List column; inner dtype must be `Float32` or
    /// `Float64` (anything else raises, ints included). `label` names
    /// the function and parameter for error messages.
    pub(crate) fn float(
        s: &Series,
        label: &str,
    ) -> PolarsResult<Self> {
        let inner_dtype = list_inner_dtype(s, label)?;
        polars_ensure!(
            matches!(inner_dtype, DataType::Float32 | DataType::Float64),
            ComputeError:
            "{label} must be a List column with Float32 or Float64 inner dtype, got {}",
            s.dtype()
        );
        let cast = s.cast(&DataType::List(Box::new(DataType::Float64)))?;
        let mut rows = Vec::with_capacity(cast.len());
        for opt in cast.list()?.into_iter() {
            rows.push(match opt {
                None => None,
                Some(row) => Some(row.f64()?.into_iter().collect()),
            });
        }
        Ok(Self { rows, inner_dtype })
    }
}

impl TypedListInput<bool> {
    /// Extract a Boolean List column; any other inner dtype raises.
    pub(crate) fn boolean(
        s: &Series,
        label: &str,
    ) -> PolarsResult<Self> {
        let inner_dtype = list_inner_dtype(s, label)?;
        polars_ensure!(
            inner_dtype == DataType::Boolean,
            ComputeError: "{label} must be a List column with Boolean inner dtype, got {}",
            s.dtype()
        );
        let mut rows = Vec::with_capacity(s.len());
        for opt in s.list()?.into_iter() {
            rows.push(match opt {
                None => None,
                Some(row) => Some(row.bool()?.into_iter().collect()),
            });
        }
        Ok(Self { rows, inner_dtype })
    }
}

impl<T> TypedListInput<T> {
    /// Physical row count, before broadcasting.
    pub(crate) fn n_rows(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn inner_dtype(&self) -> &DataType {
        &self.inner_dtype
    }

    /// Row `i` with length-1 literal broadcasting; `None` is a null row.
    pub(crate) fn row(
        &self,
        i: usize,
    ) -> Option<&[Option<T>]> {
        self.rows[if self.rows.len() == 1 { 0 } else { i }].as_deref()
    }
}

/// The inner dtype of a `List` column, or a clear error.
fn list_inner_dtype(
    s: &Series,
    label: &str,
) -> PolarsResult<DataType> {
    match s.dtype() {
        DataType::List(inner) => Ok(*inner.clone()),
        dt => polars_bail!(ComputeError: "{label} must be a List column, got {dt}"),
    }
}

/// Build a `List` float series from per-row optional elements, casting
/// the `f64`-computed values to `inner` once at the exit boundary.
pub(crate) fn build_float_list(
    rows: Vec<Option<Vec<Option<f64>>>>,
    inner: &DataType,
) -> PolarsResult<Series> {
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        PlSmallStr::from(""),
        rows.len(),
        rows.len() * 8,
        DataType::Float64,
    );
    for row in rows {
        match row {
            Some(vals) => builder.append_iter(vals.into_iter()),
            None => builder.append_null(),
        }
    }
    let s = builder.finish().into_series();
    if *inner == DataType::Float64 {
        Ok(s)
    } else {
        s.cast(&DataType::List(Box::new(inner.clone())))
    }
}

/// Build a `List(Boolean)` series from per-row optional elements.
pub(crate) fn build_bool_list(rows: Vec<Option<Vec<Option<bool>>>>) -> Series {
    let mut builder =
        ListBooleanChunkedBuilder::new(PlSmallStr::from(""), rows.len(), rows.len() * 8);
    for row in rows {
        match row {
            Some(vals) => builder.append_iter(vals.into_iter()),
            None => builder.append_null(),
        }
    }
    builder.finish().into_series()
}

/// Build a `List(UInt32)` series from per-row optional elements
/// (the `count` aggregation's dtype, as in polars).
pub(crate) fn build_u32_list(rows: Vec<Option<Vec<Option<u32>>>>) -> Series {
    let mut builder = ListPrimitiveChunkedBuilder::<UInt32Type>::new(
        PlSmallStr::from(""),
        rows.len(),
        rows.len() * 8,
        DataType::UInt32,
    );
    for row in rows {
        match row {
            Some(vals) => builder.append_iter(vals.into_iter()),
            None => builder.append_null(),
        }
    }
    builder.finish().into_series()
}
