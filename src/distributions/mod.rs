use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, StudentsT};

type ChiSquaredDistributionGenerator = fn(dof: usize) -> ChiSquared;
type TDistributionGenerator = fn(dof: usize) -> StudentsT;
type FDistributionGenerator = fn(ndof: usize, ddof: usize) -> FisherSnedecor;

/// Chi-squared distribution with `dof > 0` degrees of freedom.
const CHI_SQUARED: ChiSquaredDistributionGenerator = |dof| 
-> ChiSquared { 
    ChiSquared::new(dof as f64).unwrap() 
};

/// StudentsT distribution with `dof > 0` degrees of freedom.
const T_DISTRIBUTION: TDistributionGenerator = |dof| 
-> StudentsT { 
    StudentsT::new(0.0, 1.0, dof as f64).unwrap() 
};

/// Fischer distrubution with `ndof > 0` numerator degrees of freedom and 
/// `ddof > 0` denominator degrees of freedom.
const F_DISTRIBUTION: FDistributionGenerator = |ndof, ddof| 
-> FisherSnedecor { 
    assert!(ndof > 0, "Invalid numerator degrees of Freedom in the F_DISTRIBUTION");
    assert!(ddof > 0, "Invalid denominator degrees of Freedom in the F_DISTRIBUTION");
    FisherSnedecor::new(ndof as f64, ddof as f64).unwrap() 
};

/// Calculates the cumulative distribution function for the chi-squared 
/// distribution with `dof > 0` degrees of freedom at `test_val`.
pub fn pchisq(test_val: f64, dof: usize) -> f64 {
    CHI_SQUARED(dof).cdf(test_val)
}

pub fn qchisq(test_val: f64, dof: usize) -> f64 {
    CHI_SQUARED(dof).inverse_cdf(test_val)
}

/// Calculates the cumulative distribution function for the T-distribution with
/// `dof > 0` degrees of freedom at `t_val`.
pub fn pt(t_val: f64, dof: usize) -> f64 {
    T_DISTRIBUTION(dof).cdf(t_val) 
}

/// Calculates the inverse cumulative distribution function for the 
/// T-distribution with `dof > 0` degrees of freedom at `p_val`.
pub fn qt(p_val: f64, dof: usize) -> f64 {
    T_DISTRIBUTION(dof).inverse_cdf(p_val) 
}

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
