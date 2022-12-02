use crate::{traits::number::Number, linear_model::{least_squares_estimate::LeastSquaresEstimate, added_variable_analysis::AddedVariable}, summary::fmt_number};

use super::Summary;

impl<T: Number> Summary for LeastSquaresEstimate<T> {
    fn summary(&self) -> () {
        let num_coefficients = self.coefficients.len();
        print!("\nCoefficients\t Std. Errs\t t-vals\t Pr(>|t|) \n");
        for i in 0..num_coefficients {
            print!("x{}: {:}\t {:}\t {:.2}\t {:} \n",
             i, 
             fmt_number(self.coefficients[i], 0, 3, 2),
             fmt_number(self.standard_errors[i], 0, 3, 2), 
             self.t_values[i],
             fmt_number(self.prob_t[i], 0, 3, 2)
            );
        }
        print!("\nResidual standard error {:.3}\n", 
        self.residual_standard_error,
        );
        print!("R-squared: {:.3}\t Adjusted R-squared: {:.3}\n", 
        self.r_squared,
        self.r_squared_adjusted
        );
        print!("F-statistic: {:.3} on {} and {} dof, p-value: {}\n", 
        self.f_statistic.value,
        self.f_statistic.numerator_dof,
        self.f_statistic.denominator_dof,
        fmt_number(self.f_statistic.p_value, 0, 3, 2)
        );
    }
}

impl<T: Number> Summary for AddedVariable<T> {
    fn summary(&self) -> () {
        todo!()
    }
}