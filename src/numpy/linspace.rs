/// Creates an array of evenly spaced values between `start` and `stop`.
///
/// # Arguments
///
/// * `start` - The starting value of the sequence.
/// * `stop` - The end value of the sequence, unless `endpoint` is set to false.
/// * `num` - Number of samples to generate. Default is 50.
/// * `endpoint` - If true, `stop` is the last sample. Otherwise, it is not included. Default is true.
/// * `retstep` - If true, return (`samples`, `step`), where `step` is the spacing between samples.
///
/// # Returns
///
/// If `retstep` is false, returns a vector of evenly spaced samples.
/// If `retstep` is true, returns a tuple containing the vector and the step size.
///
/// # Examples
///
/// ```
/// let samples = linspace(0.0, 10.0, 5, true, false);
/// assert_eq!(samples, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
///
/// let (samples, step) = linspace(0.0, 10.0, 5, true, true);
/// assert_eq!(samples, vec![0.0, 2.5, 5.0, 7.5, 10.0]);
/// assert_eq!(step, 2.5);
/// ```
pub fn linspace(
    start: f64,
    stop: f64,
    num: usize,
    endpoint: bool,
    retstep: bool,
) -> (Vec<f64>, Option<f64>) {
    // Handle edge cases
    if num == 0 {
        return (Vec::new(), None);
    }

    if num == 1 {
        return (vec![start], None);
    }

    let div = if endpoint { num - 1 } else { num };
    let step = (stop - start) / div as f64;

    let mut result = Vec::with_capacity(num);

    for i in 0..num {
        let value = start + step * i as f64;
        result.push(value);
    }

    // In the non-endpoint case, we need to ensure we don't include the stop value
    if !endpoint && num > 0 {
        // The loop above will have generated values up to but not including:
        // start + step * num
        // We might need to fix numerical precision errors if we're very close to stop
        if (result.last().unwrap() - stop).abs() < f64::EPSILON {
            *result.last_mut().unwrap() = stop - step / 2.0;
        }
    }

    if retstep {
        (result, Some(step))
    } else {
        (result, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper function to compare floating point vectors
    fn vec_approx_eq(
        a: &[f64],
        b: &[f64],
    ) -> bool {
        if a.len() != b.len() {
            return false;
        }

        a.iter()
            .zip(b.iter())
            .all(|(x, y)| (x - y).abs() < f64::EPSILON * 100.0)
    }

    #[test]
    fn test_basic_usage() {
        // Basic test with endpoints included
        let expected = vec![0.0, 2.5, 5.0, 7.5, 10.0];
        let result = linspace(0.0, 10.0, 5, true, false).0;
        assert!(vec_approx_eq(&result, &expected));
    }

    #[test]
    fn test_without_endpoints() {
        // Test with endpoints excluded
        let expected = vec![0.0, 2.0, 4.0, 6.0, 8.0];
        let result = linspace(0.0, 10.0, 5, false, false).0;
        assert!(vec_approx_eq(&result, &expected));
    }

    #[test]
    fn test_with_retstep() {
        // Test return of step value
        let (result, step) = linspace(0.0, 10.0, 5, true, true);
        let expected = vec![0.0, 2.5, 5.0, 7.5, 10.0];
        assert!(vec_approx_eq(&result, &expected));
        assert_eq!(step.unwrap(), 2.5);
    }

    #[test]
    fn test_negative_range() {
        // Test with negative range
        let expected = vec![0.0, -2.5, -5.0, -7.5, -10.0];
        let result = linspace(0.0, -10.0, 5, true, false).0;
        assert!(vec_approx_eq(&result, &expected));
    }

    #[test]
    fn test_single_value() {
        // Test with num = 1
        let result = linspace(5.0, 10.0, 1, true, false).0;
        assert_eq!(result, vec![5.0]);
    }

    #[test]
    fn test_empty() {
        // Test with num = 0
        let result = linspace(0.0, 1.0, 0, true, false).0;
        assert!(result.is_empty());
    }
}
