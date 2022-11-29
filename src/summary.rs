use crate::
    {
        least_squares_estimate::LeastSquaresEstimate, 
        added_variable_analysis::AddedVariable
    };

fn fmt_f64(num: f64, width: usize, precision: usize, exp_pad: usize) -> String {
    let mut num = format!("{:.precision$e}", num, precision = precision);
    let exp = num.split_off(num.find('e').unwrap());

    let (sign, exp) = if exp.starts_with("e-") {
        ('-', &exp[2..])
    } else {
        ('+', &exp[1..])
    };
    num.push_str(&format!("e{}{:0>pad$}", sign, exp, pad = exp_pad));

    format!("{:>width$}", num, width = width)
}

pub trait Summary {
    fn summary(&self) -> ();
}

impl Summary for LeastSquaresEstimate {
    fn summary(&self) -> () {
        let num_coefficients = self.coefficients.len();
        print!("\nCoefficients\t Std. Errs\t t-vals\t Pr(>|t|) \n");
        for i in 0..num_coefficients {
            print!("x{}: {:}\t {:}\t {:.2}\t {:} \n",
             i, 
             fmt_f64(self.coefficients[i], 0, 3, 2),
             fmt_f64(self.standard_errors[i], 0, 3, 2), 
             self.t_values[i],
             fmt_f64(self.prob_t[i], 0, 3, 2)
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
        fmt_f64(self.f_statistic.p_value, 0, 3, 2)
        );
    }
}

impl Summary for AddedVariable {
    fn summary(&self) -> () {
        todo!()
    }
}