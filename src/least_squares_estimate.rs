use na::DVector;

#[derive(Debug)]
pub struct FStatistic {
    pub denominator_dof: usize,
    pub numerator_dof: usize,
    pub p_value: f64,
    pub value: f64,
}

#[derive(Debug)]
/// The LeastSquaresEstimate is the result of fitting `RegressionData` under the
/// standard linear model assumptions.
/// 
/// Namely, $Y = X \beta + \epsilon$, where:
/// - $y \in \mathbb{R}^n$, 
/// - $X \in \mathbb{R}^{n \times p}$,
/// - $\beta \in \mathbb{R}^p$,
/// - $\epsilon \sim \mathcal{N}(0, \sigma^2 I_n)$.
pub struct LeastSquaresEstimate {
    /// The coefficients of regression.
    pub coefficients: DVector<f64>,
    /// The line fit with the coefficients of regression.
    pub fitted_line: DVector<f64>,
    /// The F-statistic computed under the null hypothesis 
    /// H<sub>0</sub>: `All of the model coefficients are zero`. 
    pub f_statistic: FStatistic,
    /// The unbiased estimator of the models variance.
    /// 
    /// Computed as the sum of square residuals divided by 
    /// (length of y - rank of x)
    pub mean_squared_error: f64,
    /// Index `i` is the `Pr(>|t|)` under the null hypothesis
    ///  H<sub>0</sub>: `coefficients[i] = 0`.
    pub prob_t: DVector<f64>,
    /// The error between the observations and the fitted line.
    pub residuals: DVector<f64>,
    /// The square root of the `mean_squared_error`.
    pub residual_standard_error: f64,
    /// Measure of how well linear model fits data. 
    /// 
    /// Alternatively viewed as the square of the correlation between 
    /// `y` and the `fitted_line`.
    /// 
    /// ### Notes
    /// Use this and `r_squared_adjusted` appropriately to determine
    /// model goodness. See 
    /// [this answer by whuber][https://stats.stackexchange.com/questions/13314/is-r2-useful-or-dangerous].
    pub r_squared: f64,
    /// - `r_squared` is trying to estimate a fractional reduction in variance
    ///  but is using biased quantities. 
    /// - `r_squared_adjusted` accounts for the appropriate degrees of freedom
    ///  adjustments and imposes a higher penalty for a higher rank model.
    /// 
    /// ### Notes
    /// - `r_squared_adjusted` can be negative. Use this and `r_squared`
    ///  appropriately to determine model goodness. See 
    /// [this answer by whuber][https://stats.stackexchange.com/questions/13314/is-r2-useful-or-dangerous].
    pub r_squared_adjusted: f64,
    /// Index `i` is `coefficients[i] / x[i].norm()`. 
    pub standard_errors: DVector<f64>,
    /// Index `i` is `coefficients[i] / standard_errors[i]`.
    pub t_values: DVector<f64>,
}

impl LeastSquaresEstimate {
    pub fn predict(&self, new_x: DVector<f64>) -> f64 {
        self.coefficients.dot(&new_x)
    }
}
