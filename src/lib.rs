mod agg;
mod dsp;
mod dsp_util;
mod feat;
mod op;
mod util;
use dsp_util::{fft_freqs, fft_freqs_linspace};
use {pyo3::prelude::*, pyo3_polars::PolarsAllocator};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pymodule]
fn _internal(
    _py: Python,
    m: &Bound<PyModule>,
) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(fft_freqs, m)?)?;
    m.add_function(wrap_pyfunction!(fft_freqs_linspace, m)?)?;
    Ok(())
}
