use std::ops::Mul;

use nalgebra::{DMatrix, DVector};

use crate::{traits::number::Number, dataset::Dataset};
use super::{
    least_squares_estimate::{LeastSquaresEstimate, FStatistic}, 
    added_variable_analysis::AddedVariable, 
    hypothesis_testing::{one_sided_f_test, two_sided_t_test}
};


const TOLERANCE: f64 = 1e-12;

pub trait LinearModel<T> where T: Number {
    /// Regress `y` on `x` to produce the least squares estimate of the coefficients. 
    fn fit_lm(&self) -> LeastSquaresEstimate<T>;
    /// Analyze the added variable effect of column `i` in the regression of `y` on `x`
    /// without an intercept.
    fn added_variable_analysis(&self, variable_to_analyze: usize) -> AddedVariable<T>;
    /// Return an array of analyzed added variable effects, with index `i` 
    /// being the added variable effect of column `i`.
    fn added_variable_analysis_all(&self) -> Vec<AddedVariable<T>>;
}

fn extract_column<T: Number>(matrix: &DMatrix<T>, i: usize) -> DVector<T> {
    DVector::from_column_slice(matrix.column(i).clone_owned().as_slice())
}

/// Computes the Moore-Penrose inverse of matrix
fn compute_pseudo_inverse<T: Number>(matrix: &DMatrix<T>) -> DMatrix<T> {
    // TODO: safely handle error
    return matrix.clone_owned().pseudo_inverse(T::from_f64(TOLERANCE).unwrap()).unwrap(); 
}

impl<'a, T> LinearModel<T> for Dataset<'a, T> where T:Number 
{
    fn added_variable_analysis(&self, variable_to_analyze: usize) -> AddedVariable<T> {
        assert!((0..self.features.shape().1).contains(&variable_to_analyze));
        let y = self.labels.clone_owned();
        let (num_rows, num_cols) = self.features.shape();

        let x_i: DVector<T> = extract_column(&self.features, variable_to_analyze);
        let w: DMatrix<T> = self.features.clone_owned().remove_column(variable_to_analyze);

        let projection_onto_w: DMatrix<T> = &w * compute_pseudo_inverse(&w);

        let x_tilde: DVector<T> = &x_i - &projection_onto_w * &x_i;
        let y_tilde: DVector<T> = &y - projection_onto_w.mul(&y);

        let v_top: T = x_i.add_scalar(T::one().neg() * x_i.mean()).norm_squared();
        let v_simple: T = x_tilde.norm_squared();
        let variance_inflation_factor: T = v_top / v_simple;

        let coefficient: T = y.dot(&x_tilde) / x_tilde.norm_squared();
        let residuals: DVector<T> = &y_tilde - &x_tilde * coefficient;
        let r_i: DVector<T> = &residuals + &x_i * coefficient;

        let sse: T = residuals.norm_squared();

        let rank: usize = self.features.rank(T::from_f64(TOLERANCE).unwrap());
        let mean_squared_error: T = sse / (T::from_usize(num_rows).unwrap() - T::from_usize(rank).unwrap());

        let t_squared: T = coefficient.powf(T::from_usize(2).unwrap()) / mean_squared_error / x_i.norm_squared();
        let squared_correlation_prp: T = t_squared * variance_inflation_factor;
        let squared_correlation_avp: T = squared_correlation_prp / (T::from_usize(num_rows).unwrap() - T::from_usize(num_cols).unwrap() + squared_correlation_prp);

        AddedVariable { 
            coefficient,
            partial_residual: r_i,
            squared_correlation_avp,
            squared_correlation_prp,
            standard_error: T::zero(),
            variance_inflation_factor, 
            x_tilde, 
            y_tilde 
        }
    }

    fn added_variable_analysis_all(&self) -> Vec<AddedVariable<T>> {
        let num_cols = self.features.shape().1;
        (0..num_cols)
            .map(|variable_to_analyze: usize| self.added_variable_analysis(variable_to_analyze))
            .collect::<Vec<AddedVariable<T>>>()
    }


    fn fit_lm(&self) -> LeastSquaresEstimate<T> {
        let x: &DMatrix<T> = &self.features;
        let y: &DVector<T> = &self.labels;

        let mut col1: DVector<T> = x.column(0).clone_owned();
        col1.fill(T::one());
        let has_intercept: bool = x.column(0).eq(&col1);

        let xtx_inv: DMatrix<T> = (x.transpose() * x).try_inverse().unwrap();
        let pinv: DMatrix<T> = &xtx_inv * x.transpose(); 
        let coefficients: DVector<T> = &pinv * y;
        let fitted_line: DVector<T> = x * &coefficients;
    
        let (num_rows, _) = x.shape();
        let rank: usize = x.rank(T::from_f64(TOLERANCE).unwrap());
        let dof: usize = num_rows - rank;
    
        let residuals: DVector<T> = y - &fitted_line;
        let sse: T = residuals.norm_squared();
        let mean_squared_error: T = sse / T::from_usize(dof).unwrap();
        let residual_standard_error: T = mean_squared_error.sqrt();

        let (y_mean, adj_r_squared_dof, ndof) = match has_intercept {
            true => (y.mean(), num_rows - 1, rank - 1),
            false => (T::zero(), num_rows, rank) 
         };

        let ssy: T = y.add_scalar(T::one().neg() * y_mean).norm_squared();
        
        let r_squared: T = T::one() - (sse / ssy);
        let r_squared_adjusted: T = T::one() - (mean_squared_error / (ssy / T::from_usize(adj_r_squared_dof).unwrap()));
    
        let f_stat: T = r_squared / (T::one() - r_squared) * (T::from_usize(dof).unwrap() / T::from_usize(ndof).unwrap());
        let p_value: T =  T::from_f64(one_sided_f_test(T::as_(f_stat), ndof, dof)).unwrap();
    
        let standard_errors: DVector<T> = xtx_inv.diagonal().map(|v| v.sqrt()) * residual_standard_error;
        let t_values: DVector<T> = coefficients.component_div(&standard_errors);
        let prob_t: DVector<T> = t_values.map(|t| T::from_f64(two_sided_t_test(T::as_(t), dof)).unwrap());
        
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
