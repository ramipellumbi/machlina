use na::DVector;

#[derive(Debug)]
/// The added variable effect and partial residual
/// of column `i` in the regression. 
/// 
/// #### Notes
/// The residuals as given by
/// ```
/// let data = RegressionData{ x: x_tilde, y: y_tilde };
/// let lm = data.fit();
/// let residuals = lm.residuals;
/// ```
/// are the same residuals as that of the full model.
pub struct AddedVariable {
    /// The coefficient of regression of `x_i` in the full regression of `y` on `x`.
    /// ### Notes
    /// The variance of the coefficient of regression of `x_i` in the full model is given by
    /// `var / x_tilde.norm_squared()`.
    pub coefficient: f64,
    pub partial_residual: DVector<f64>,
    /// TODO: DO NOT STORE IN EVERY INDEX OF THE ALL ANALYSIS
    pub residuals: DVector<f64>,
    /// TODO: DOCUMENT ME
    pub squared_correlation_avp: f64,
    pub squared_correlation_prp: f64,
    /// The ratio of the variance of the coefficient of `x_i` using the full model to the variance of the coefficient
    /// had we just regressed `y` on `x_i`.
    /// 
    /// ### Notes
    /// `1 / variance_inflation_factor` is the fraction of `var(x_i)` NOT explained by regression of `x_i` on `[x_{1}, ..., x_{i-1}, x_{i+1}, ... x_p]`.
    pub variance_inflation_factor: f64,
    /// The residual between `x_i` and it's projection onto the column span of `[x_{1}, ..., x_{i-1}, x_{i+1}, ... x_p]`.
    pub x_tilde: DVector<f64>,
    /// The residuals between y and y projected onto the column span of `[x_{1}, ..., x_{i-1}, x_{i+1}, ... x_p]`.
    pub y_tilde: DVector<f64>,
}
