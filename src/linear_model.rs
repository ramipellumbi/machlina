use na::{DMatrix, DVector};

use crate::added_variable_analysis::AddedVariable;
use crate::least_squares_estimate::{LeastSquaresEstimate, FStatistic};
use crate::regression_data::RegressionData;
use crate::hypothesis_testing::{one_sided_f_test, two_sided_t_test};

const TOLERANCE: f64 = 1e-12;

pub trait LinearModel {
    /// Regress `y` on `x` to produce the least squares estimate of the coefficients. 
    fn fit_lm(&self) -> LeastSquaresEstimate;
    /// Analyze the added variable effect of column `i` in the regression of `y` on `x`
    /// without an intercept.
    fn added_variable_analysis(&self, variable_to_analyze: usize) -> AddedVariable;
    /// Return an array of analyzed added variable effects, with index `i` 
    /// being the added variable effect of column `i`.
    fn added_variable_analysis_all(&self) -> Vec<AddedVariable>;
}

fn extract_column(matrix: &DMatrix<f64>, i: usize) -> DVector<f64> {
    DVector::from_column_slice(matrix.column(i).clone_owned().as_slice())
}

/// Computes the Moore-Penrose inverse of matrix
fn compute_pseudo_inverse(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    // TODO: safely handle error
    return matrix.clone_owned().pseudo_inverse(TOLERANCE).unwrap(); 
}

impl LinearModel for RegressionData 
{
    fn added_variable_analysis(&self, variable_to_analyze: usize) -> AddedVariable {
        assert!((0..self.x.shape().1).contains(&variable_to_analyze));
        let (num_rows, num_cols) = self.x.shape();

        let x_i: DVector<f64> = extract_column(&self.x, variable_to_analyze);
        let w: DMatrix<f64> = self.x.clone_owned().remove_column(variable_to_analyze);

        let projection_onto_w: DMatrix<f64> = &w * compute_pseudo_inverse(&w);

        let x_tilde: DVector<f64> = &x_i - &projection_onto_w * &x_i;
        let y_tilde: DVector<f64> = &self.y - &projection_onto_w * &self.y;

        let v_top: f64 = x_i.add_scalar(-1.0 * x_i.mean()).norm_squared();
        let v_simple: f64 = x_tilde.norm_squared();
        let variance_inflation_factor: f64 = v_top / v_simple;

        let coefficient: f64 = self.y.dot(&x_tilde) / x_tilde.norm_squared();
        let residuals: DVector<f64> = &y_tilde - &x_tilde * coefficient;
        let r_i: DVector<f64> = &residuals + coefficient * &x_i;

        let sse: f64 = residuals.norm_squared();

        let rank: usize = self.x.rank(TOLERANCE);
        let mean_squared_error: f64 = sse / (num_rows - rank) as f64;

        let t_squared: f64 = coefficient.powf(2.0) / mean_squared_error / x_i.norm_squared();
        let squared_correlation_prp: f64 = t_squared * variance_inflation_factor;
        let squared_correlation_avp: f64 = squared_correlation_prp / (num_rows as f64 - num_cols as f64 + squared_correlation_prp);

        AddedVariable { 
            coefficient,
            partial_residual: r_i,
            residuals,
            squared_correlation_avp,
            squared_correlation_prp,
            variance_inflation_factor, 
            x_tilde, 
            y_tilde 
        }
    }

    fn added_variable_analysis_all(&self) -> Vec<AddedVariable> {
        let num_cols = self.x.shape().1;
        (0..num_cols)
            .map(|variable_to_analyze: usize| self.added_variable_analysis(variable_to_analyze))
            .collect::<Vec<AddedVariable>>()
    }

    fn fit_lm(&self) -> LeastSquaresEstimate {
        let x: &DMatrix<f64> = &self.x;
        let y: &DVector<f64> = &self.y;

        let mut col1: DVector<f64> = x.column(0).clone_owned();
        col1.fill(1.0);
        let has_intercept: bool = x.column(0).eq(&col1);

        let xtx_inv: DMatrix<f64> = (x.transpose() * x).try_inverse().unwrap();
        let pinv: DMatrix<f64> = &xtx_inv * x.transpose(); 
        let coefficients: DVector<f64> = &pinv * y;
        let fitted_line: DVector<f64> = x * &coefficients;
    
        let (num_rows, _) = x.shape();
        let rank: usize = x.rank(TOLERANCE);
        let dof: usize = num_rows - rank;
    
        let residuals: DVector<f64> = y - &fitted_line;
        let sse: f64 = residuals.norm_squared();
        let mean_squared_error: f64 = sse / dof as f64;
        let residual_standard_error: f64 = mean_squared_error.sqrt();

        let (y_mean, adj_r_squared_dof, ndof) = match has_intercept {
            true => (y.mean(), num_rows - 1, rank - 1),
            false => (0.0, num_rows, rank) 
         };

        let ssy: f64 = y.add_scalar(-1.0 * y_mean).norm_squared();
        
        let r_squared: f64 = 1.0 - (sse / ssy);
        let r_squared_adjusted: f64 = 1.0 - (mean_squared_error / (ssy / adj_r_squared_dof as f64));
    
        let f_stat: f64 = r_squared / (1.0 - r_squared) * (dof as f64 / ndof as f64);
        let p_value: f64 =  one_sided_f_test(f_stat, ndof, dof);
    
        let standard_errors: DVector<f64> = xtx_inv.diagonal().map(|v| v.sqrt()) * residual_standard_error;
        let t_values: DVector<f64> = coefficients.component_div(&standard_errors);
        let prob_t: DVector<f64> = t_values.map(|t| two_sided_t_test(t, dof));
        
        LeastSquaresEstimate { 
            coefficients, 
            fitted_line, 
            f_statistic: FStatistic { value: f_stat, numerator_dof: ndof, denominator_dof: dof, p_value }, 
            mean_squared_error, 
            prob_t,
            residual_standard_error, 
            residuals, 
            r_squared,
            r_squared_adjusted,
            standard_errors,
            t_values
        }
    }
}
