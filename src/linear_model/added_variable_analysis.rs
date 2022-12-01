use nalgebra::DVector;

use crate::traits::number::Number;

#[derive(Debug)]
/// The added variable effect and partial residual
/// of column `i` in the regression. 
pub struct AddedVariable<T> where T: Number {
    /// The coefficient of regression of `x_i` in the full regression of `y` on `x`.
    /// ### Notes
    /// The variance of the coefficient of regression of `x_i` in the full model is given by
    /// `var / x_tilde.norm_squared()`.
    pub coefficient: T,
    pub partial_residual: DVector<T>,
    /// TODO: DOCUMENT ME
    pub squared_correlation_avp: T,
    pub squared_correlation_prp: T,
    pub standard_error: T,
    /// The ratio of the variance of the coefficient of `x_i` using the full model to the variance of the coefficient
    /// had we just regressed `y` on `x_i`.
    /// 
    /// #### Notes
    /// `1 / variance_inflation_factor` is the fraction of `var(x_i)` NOT explained by 
    /// regression of `x_i` on `[x_{1}, ..., x_{i-1}, x_{i+1}, ... x_p]`.
    pub variance_inflation_factor: T,
    /// The residual between `x_i` and it's projection onto the column span of `[x_{1}, ..., x_{i-1}, x_{i+1}, ... x_p]`.
    pub x_tilde: DVector<T>,
    /// The residuals between y and y projected onto the column span of `[x_{1}, ..., x_{i-1}, x_{i+1}, ... x_p]`.
    pub y_tilde: DVector<T>,
}

impl<T> AddedVariable<T> where T: Number {
    /// Residuals of `y_tilde` regressed on `x–tilde`.
    /// #### Notes
    /// The residuals are the same residuals as that of the full model `y` regressed on `x`.
    pub fn residuals(&self) -> DVector<T> {
       &self.y_tilde - &self.x_tilde * self.coefficient 
    }
}
