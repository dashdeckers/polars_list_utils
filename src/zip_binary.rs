use crate::util::{
    Container, TypedListInput, broadcast_len, build_bool_list, build_float_list, tot_cmp,
};
use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::cmp::Ordering;

/// Element-wise binary operations between two paired list columns.
///
/// Arithmetic ops take float operands and keep a float output;
/// `and`/`or` take Boolean operands (Kleene logic); the comparisons
/// take float operands and emit Boolean.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    And,
    Or,
    Gt,
    Ge,
    Lt,
    Le,
    Eq,
    Ne,
}

impl Op {
    fn takes_booleans(self) -> bool {
        matches!(self, Op::And | Op::Or)
    }

    fn is_arithmetic(self) -> bool {
        matches!(self, Op::Add | Op::Sub | Op::Mul | Op::Div)
    }
}

#[derive(Deserialize)]
struct ZipBinaryKwargs {
    op: Op,
}

/// Per element, identical to the corresponding scalar polars op:
/// null propagates, NaN follows IEEE float arithmetic.
fn arithmetic(
    op: Op,
    a: Option<f64>,
    b: Option<f64>,
) -> Option<f64> {
    let (a, b) = (a?, b?);
    Some(match op {
        Op::Add => a + b,
        Op::Sub => a - b,
        Op::Mul => a * b,
        Op::Div => a / b,
        _ => unreachable!("arithmetic called with non-arithmetic op"),
    })
}

/// Comparisons follow polars' total order (NaN equals NaN and exceeds
/// everything else); null operands compare to null.
fn comparison(
    op: Op,
    a: Option<f64>,
    b: Option<f64>,
) -> Option<bool> {
    let ord = tot_cmp(a?, b?);
    Some(match op {
        Op::Gt => ord == Ordering::Greater,
        Op::Ge => ord != Ordering::Less,
        Op::Lt => ord == Ordering::Less,
        Op::Le => ord != Ordering::Greater,
        Op::Eq => ord == Ordering::Equal,
        Op::Ne => ord != Ordering::Equal,
        _ => unreachable!("comparison called with non-comparison op"),
    })
}

/// Kleene three-valued logic, as in polars: `False & null = False`,
/// `True | null = True`, otherwise null wins.
fn kleene(
    op: Op,
    a: Option<bool>,
    b: Option<bool>,
) -> Option<bool> {
    match op {
        Op::And => match (a, b) {
            (Some(false), _) | (_, Some(false)) => Some(false),
            (Some(true), Some(true)) => Some(true),
            _ => None,
        },
        Op::Or => match (a, b) {
            (Some(true), _) | (_, Some(true)) => Some(true),
            (Some(false), Some(false)) => Some(false),
            _ => None,
        },
        _ => unreachable!("kleene called with non-boolean op"),
    }
}

/// Zip one row's element pairs through `f`; a null row on either side
/// stays null, mismatched lengths raise, empty lists zip to empty.
fn zip_row<T: Copy, V>(
    left: Option<&[Option<T>]>,
    right: Option<&[Option<T>]>,
    f: impl Fn(Option<T>, Option<T>) -> Option<V>,
) -> PolarsResult<Option<Vec<Option<V>>>> {
    let (Some(left), Some(right)) = (left, right) else {
        return Ok(None);
    };
    // No row index: plugins are handed one chunk at a time, so any
    // position we could name here is chunk-local and would point at a
    // different, innocent row of the frame.
    polars_ensure!(
        left.len() == right.len(),
        ComputeError:
        "zip_binary: a row's left and right lists differ in length ({} vs {})",
        left.len(), right.len()
    );
    Ok(Some(
        left.iter().zip(right).map(|(&a, &b)| f(a, b)).collect(),
    ))
}

/// The arithmetic output inner dtype follows polars supertyping:
/// `Float32` only when both operands are `Float32`.
fn arithmetic_inner(
    left: &DataType,
    right: &DataType,
) -> DataType {
    if *left == DataType::Float32 && *right == DataType::Float32 {
        DataType::Float32
    } else {
        DataType::Float64
    }
}

/// Plan-time container and dtype validation for one operand.
fn validate_operand(
    dtype: &DataType,
    op: Op,
    label: &str,
) -> PolarsResult<(Container, DataType)> {
    let (container, inner) = Container::split(dtype, label)?;
    if op.takes_booleans() {
        polars_ensure!(
            inner == DataType::Boolean,
            ComputeError: "{label} must have a Boolean inner dtype for 'and'/'or', \
            got {dtype}"
        );
    } else {
        polars_ensure!(
            matches!(inner, DataType::Float32 | DataType::Float64),
            ComputeError: "{label} must have a Float32 or Float64 inner dtype, got {dtype}"
        );
    }
    Ok((container, inner))
}

/// The output container: `Array` only when both operands are `Array`, in
/// which case their widths must agree — a mismatch the schema already
/// knows about, so it raises at plan time. A mixed `List`/`Array` pair is
/// allowed and yields `List`, with the length checked per row.
fn zip_container(
    left: Container,
    right: Container,
) -> PolarsResult<Container> {
    match (left, right) {
        (Container::Array(lw), Container::Array(rw)) => {
            polars_ensure!(
                lw == rw,
                ComputeError:
                "zip_binary: Array operands must have equal widths, got {lw} and {rw}"
            );
            Ok(Container::Array(lw))
        }
        _ => Ok(Container::List),
    }
}

fn zip_binary_output(
    input_fields: &[Field],
    kwargs: ZipBinaryKwargs,
) -> PolarsResult<Field> {
    polars_ensure!(
        input_fields.len() == 2,
        ComputeError: "zip_binary expects exactly 2 inputs, got {}",
        input_fields.len()
    );
    let (lc, left) = validate_operand(
        input_fields[0].dtype(),
        kwargs.op,
        "zip_binary: left_column",
    )?;
    let (rc, right) = validate_operand(
        input_fields[1].dtype(),
        kwargs.op,
        "zip_binary: right_column",
    )?;
    let inner = if kwargs.op.is_arithmetic() {
        arithmetic_inner(&left, &right)
    } else {
        DataType::Boolean
    };
    Ok(Field::new(
        PlSmallStr::from(""),
        zip_container(lc, rc)?.dtype(inner),
    ))
}

/// Apply a binary operation element-wise between two paired list
/// columns.
///
/// Per element the result is identical to the corresponding scalar
/// polars op (null propagation, Kleene logic for `and`/`or`,
/// total-ordered comparisons). A null row on either side yields a null
/// row; empty lists zip to empty lists; mismatched lengths raise.
#[polars_expr(output_type_func_with_kwargs=zip_binary_output)]
fn zip_binary(
    inputs: &[Series],
    kwargs: ZipBinaryKwargs,
) -> PolarsResult<Series> {
    polars_ensure!(
        inputs.len() == 2,
        ComputeError: "zip_binary expects exactly 2 inputs, got {}",
        inputs.len()
    );
    let op = kwargs.op;

    if op.takes_booleans() {
        let left =
            TypedListInput::<bool>::boolean(&inputs[0], "zip_binary: left_column")?;
        let right =
            TypedListInput::<bool>::boolean(&inputs[1], "zip_binary: right_column")?;
        let out = zip_container(left.container(), right.container())?;
        let len = broadcast_len(&[left.n_rows(), right.n_rows()])?;
        let rows = (0..len)
            .map(|i| zip_row(left.row(i), right.row(i), |a, b| kleene(op, a, b)))
            .collect::<PolarsResult<Vec<_>>>()?;
        return build_bool_list(rows, out);
    }

    let left = TypedListInput::<f64>::float(&inputs[0], "zip_binary: left_column")?;
    let right = TypedListInput::<f64>::float(&inputs[1], "zip_binary: right_column")?;
    let out = zip_container(left.container(), right.container())?;
    let len = broadcast_len(&[left.n_rows(), right.n_rows()])?;

    if op.is_arithmetic() {
        let rows = (0..len)
            .map(|i| zip_row(left.row(i), right.row(i), |a, b| arithmetic(op, a, b)))
            .collect::<PolarsResult<Vec<_>>>()?;
        build_float_list(
            rows,
            &arithmetic_inner(left.inner_dtype(), right.inner_dtype()),
            out,
        )
    } else {
        let rows = (0..len)
            .map(|i| zip_row(left.row(i), right.row(i), |a, b| comparison(op, a, b)))
            .collect::<PolarsResult<Vec<_>>>()?;
        build_bool_list(rows, out)
    }
}
