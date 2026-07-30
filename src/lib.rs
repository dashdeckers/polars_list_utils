mod agg_slices;
mod apply_butterworth;
mod apply_fft;
mod apply_interp;
mod filter;
mod slice;
mod statistic;
mod transform;
mod util;

use {pyo3::prelude::*, pyo3_polars::PolarsAllocator};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(
    _py: Python,
    m: &Bound<PyModule>,
) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
