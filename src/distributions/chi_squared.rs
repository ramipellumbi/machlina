use statrs::distribution::{ChiSquared, ContinuousCDF};

type ChiSquaredDistributionGenerator = fn(dof: usize) -> ChiSquared;

/// Chi-squared distribution with `dof > 0` degrees of freedom.
const CHI_SQUARED: ChiSquaredDistributionGenerator = |dof| 
-> ChiSquared { 
    assert!(dof > 0, "Must have greater than zero degrees of freedom");
    ChiSquared::new(dof as f64).unwrap() 
};

/// Calculates the cumulative distribution function for the chi-squared 
/// distribution with `dof > 0` degrees of freedom at `test_val`.
pub fn pchisq(test_val: f64, dof: usize) -> f64 {
    CHI_SQUARED(dof).cdf(test_val)
}

pub fn qchisq(test_val: f64, dof: usize) -> f64 {
    CHI_SQUARED(dof).inverse_cdf(test_val)
}
