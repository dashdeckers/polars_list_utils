/// Implements NumPy's np.interp function in pure Rust
///
/// Performs one-dimensional linear interpolation.
///
/// # Arguments
/// * `x` - The x-coordinates at which to evaluate the interpolated values
/// * `xp` - The x-coordinates of the data points
/// * `fp` - The y-coordinates of the data points
/// * `left` - Value to return for x < xp[0], default is fp[0]
/// * `right` - Value to return for x > xp[-1], default is fp[-1]
/// * `period` - A period for the x-coordinates. This parameter allows making the interpolation periodic.
///   If specified, values of x outside the range xp[0] to xp[-1] are mapped into that range
///   by periodic extension. Default is None.
///
/// # Returns
/// The interpolated values, same shape as x.
///
/// # Examples
/// ```
/// let x = vec![0.0, 1.0, 1.5, 2.5, 3.5];
/// let xp = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let fp = vec![10.0, 20.0, 30.0, 40.0, 50.0];
///
/// let result = interp(&x, &xp, &fp, None, None, None);
/// assert_eq!(result, vec![10.0, 10.0, 15.0, 25.0, 35.0]);
/// ```
pub fn interp(
    x: &[f64],
    xp: &[f64],
    fp: &[f64],
    left: Option<f64>,
    right: Option<f64>,
    period: Option<f64>,
) -> Vec<f64> {
    if xp.is_empty() {
        panic!("xp must not be empty");
    }
    if xp.len() != fp.len() {
        panic!("xp and fp must have the same length");
    }

    // Create output vector of the same length as x
    let mut result = Vec::with_capacity(x.len());

    // Handle the case where xp has only one element
    if xp.len() == 1 {
        let fill_value = fp[0];
        for _ in 0..x.len() {
            result.push(fill_value);
        }
        return result;
    }

    // Determine if the array is monotonically increasing
    let is_increasing = xp.windows(2).all(|w| w[0] <= w[1]);

    // If not monotonically increasing, sort the data points
    let (xp_sorted, fp_sorted) = if !is_increasing {
        let mut sorted_pairs: Vec<(f64, f64)> =
            xp.iter().cloned().zip(fp.iter().cloned()).collect();
        sorted_pairs
            .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut xp_s = Vec::with_capacity(xp.len());
        let mut fp_s = Vec::with_capacity(fp.len());

        for (x_val, f_val) in sorted_pairs {
            xp_s.push(x_val);
            fp_s.push(f_val);
        }
        (xp_s, fp_s)
    } else {
        (xp.to_vec(), fp.to_vec())
    };

    // Get default values for left and right extrapolation
    let left_val = left.unwrap_or(fp_sorted[0]);
    let right_val = right.unwrap_or(*fp_sorted.last().unwrap());

    for &x_val in x {
        let x_periodic = if let Some(p) = period {
            if p <= 0.0 {
                panic!("period must be positive");
            }

            let mut x_mapped = x_val;
            let x_min = xp_sorted[0];
            let x_max = *xp_sorted.last().unwrap();
            let range = x_max - x_min;

            if range <= 0.0 {
                // Can't apply periodicity if the range is 0 or negative
                x_val
            } else {
                // Map x_val to the range [x_min, x_min + period) by periodic extension
                if x_mapped < x_min || x_mapped >= x_min + p {
                    x_mapped = x_min + ((x_mapped - x_min) % p + p) % p;
                }

                // In NumPy's interp with period, if the input maps exactly to x_min,
                // or if it maps to a value greater than x_max after periodic mapping,
                // it's treated specially
                if x_mapped == x_min {
                    x_mapped = x_min; // Explicitly handle this case
                } else if x_mapped > x_max {
                    // Map values above x_max to x_max exactly
                    x_mapped = x_max;
                }

                x_mapped
            }
        } else {
            x_val
        };

        // Handle extrapolation cases
        if x_periodic < xp_sorted[0] {
            result.push(left_val);
            continue;
        } else if x_periodic > *xp_sorted.last().unwrap() {
            result.push(right_val);
            continue;
        }

        // Binary search to find the right interval
        let mut low = 0;
        let mut high = xp_sorted.len() - 1;

        while low < high {
            let mid = (low + high) / 2;
            if xp_sorted[mid] < x_periodic {
                low = mid + 1;
            } else {
                high = mid;
            }
        }

        // Adjust if we're at the right endpoint
        if low > 0 && xp_sorted[low] > x_periodic {
            low -= 1;
        }

        // Linear interpolation formula
        if xp_sorted[low] == x_periodic {
            // Exact match, no interpolation needed
            result.push(fp_sorted[low]);
        } else if low + 1 < xp_sorted.len() && xp_sorted[low + 1] == x_periodic {
            // Exact match with the next point
            result.push(fp_sorted[low + 1]);
        } else if low + 1 < xp_sorted.len() {
            // Regular interpolation
            let x_low = xp_sorted[low];
            let x_high = xp_sorted[low + 1];
            let y_low = fp_sorted[low];
            let y_high = fp_sorted[low + 1];

            let slope = (y_high - y_low) / (x_high - x_low);
            let y_interp = y_low + slope * (x_periodic - x_low);

            result.push(y_interp);
        } else {
            // We're at the last point but didn't match exactly
            result.push(right_val);
        }
    }

    result
}

/// Helper function to test the interp implementation against simple cases
#[cfg(test)]
mod tests {
    use super::interp;

    #[test]
    fn test_basic_interpolation() {
        let x = vec![0.0, 1.0, 1.5, 2.5, 3.5];
        let xp = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let fp = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let result = interp(&x, &xp, &fp, None, None, None);

        // Expected results based on NumPy behavior:
        // - x=0.0: Below xp[0], so should be fp[0] = 10.0
        // - x=1.0: Equal to xp[0], so should be fp[0] = 10.0
        // - x=1.5: Between xp[0]=1.0 and xp[1]=2.0, should be 15.0
        // - x=2.5: Between xp[1]=2.0 and xp[2]=3.0, should be 25.0
        // - x=3.5: Between xp[2]=3.0 and xp[3]=4.0, should be 35.0
        assert_eq!(result, vec![10.0, 10.0, 15.0, 25.0, 35.0]);
    }

    #[test]
    fn test_custom_bounds() {
        let x = vec![-1.0, 0.5, 6.0];
        let xp = vec![1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0, 30.0];

        let result = interp(&x, &xp, &fp, Some(-10.0), Some(100.0), None);

        // Expected results:
        // - x=-1.0: Below xp[0], so should be left = -10.0
        // - x=0.5: Below xp[0], so should be left = -10.0
        // - x=6.0: Above xp[-1], so should be right = 100.0
        assert_eq!(result, vec![-10.0, -10.0, 100.0]);
    }

    #[test]
    fn test_unsorted_input() {
        let x = vec![1.5, 2.5];
        let xp = vec![3.0, 1.0, 2.0]; // Unsorted
        let fp = vec![30.0, 10.0, 20.0]; // Corresponding to unsorted xp

        let result = interp(&x, &xp, &fp, None, None, None);

        // Expected results after sorting (xp, fp) to ([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]):
        // - x=1.5: Between sorted xp[0]=1.0 and xp[1]=2.0, should be 15.0
        // - x=2.5: Between sorted xp[1]=2.0 and xp[2]=3.0, should be 25.0
        assert_eq!(result, vec![15.0, 25.0]);
    }

    #[test]
    fn test_period() {
        let x = vec![0.0, 6.0, -2.0];
        let xp = vec![1.0, 2.0, 3.0, 4.0];
        let fp = vec![10.0, 20.0, 30.0, 40.0];

        // Period of 4.0, so values should wrap around every 4 units
        let result = interp(&x, &xp, &fp, None, None, Some(4.0));

        // Updated expected results with period=4.0:
        // - x=0.0: Maps to 4.0 (wraps to range [1.0, 5.0)), which equals
        //          xp[3]=4.0, should be 40.0
        // - x=6.0: Maps to 2.0, equals xp[1]=2.0, should be 20.0
        // - x=-2.0: Maps to 2.0, equals xp[1]=2.0, should be 20.0
        assert_eq!(result, vec![40.0, 20.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "xp must not be empty")]
    fn test_empty_xp() {
        let x = vec![1.0, 2.0];
        let xp: Vec<f64> = vec![];
        let fp: Vec<f64> = vec![];

        interp(&x, &xp, &fp, None, None, None);
    }

    #[test]
    #[should_panic(expected = "xp and fp must have the same length")]
    fn test_mismatched_length() {
        let x = vec![1.0, 2.0];
        let xp = vec![1.0, 2.0, 3.0];
        let fp = vec![10.0, 20.0];

        interp(&x, &xp, &fp, None, None, None);
    }
}
