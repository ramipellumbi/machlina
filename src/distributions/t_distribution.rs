use statrs::distribution::{ContinuousCDF, StudentsT};

type TDistributionGenerator = fn(dof: usize) -> StudentsT;

/// StudentsT distribution with `dof > 0` degrees of freedom.
const T_DISTRIBUTION: TDistributionGenerator = |dof| 
-> StudentsT { 
    assert!(dof > 0, "Must have greater than zero degrees of freedom");
    StudentsT::new(0.0, 1.0, dof as f64).unwrap() 
};

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
