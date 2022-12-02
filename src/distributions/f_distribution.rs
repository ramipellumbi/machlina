use statrs::distribution::{ContinuousCDF, FisherSnedecor};

type FDistributionGenerator = fn(ndof: usize, ddof: usize) -> FisherSnedecor;

/// Fischer distrubution with `ndof > 0` numerator degrees of freedom and 
/// `ddof > 0` denominator degrees of freedom.
const F_DISTRIBUTION: FDistributionGenerator = |ndof, ddof| 
-> FisherSnedecor { 
    assert!(ndof > 0, "Invalid numerator degrees of Freedom in the F_DISTRIBUTION");
    assert!(ddof > 0, "Invalid denominator degrees of Freedom in the F_DISTRIBUTION");
    FisherSnedecor::new(ndof as f64, ddof as f64).unwrap() 
};

/// Calculates the cumulative distribution function for the F-distribution with: 
/// - `ndof` numerator degrees of freedom
/// - `ddof` denominator degrees of freedom
/// 
/// at `f_val`. 
pub fn pf(f_val: f64, ndof: usize, ddof: usize) -> f64 {
    F_DISTRIBUTION(ndof, ddof).cdf(f_val)
}

/// Calculates the inverse cumulative distribution function for the 
/// F-distribution with: 
/// - `ndof` numerator degrees of freedom
/// - `ddof` denominator degrees of freedom
/// 
/// at `p_val`. 
pub fn qf(p_val: f64, ndof: usize, ddof: usize) -> f64 {
    F_DISTRIBUTION(ndof, ddof).inverse_cdf(p_val)
}
