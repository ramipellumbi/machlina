extern crate nalgebra as na;

pub mod added_variable_analysis;
pub mod distributions;
pub mod hypothesis_testing;
pub mod least_squares_estimate;
pub mod linear_model;
pub mod regression_data;
pub mod summary;

use na::{DMatrix, DVector};
use summary::Summary;
use crate::{regression_data::RegressionData, linear_model::LinearModel};

fn main() {
    let x: DMatrix<f64> = DMatrix::from_row_slice(8, 2, &[
        10.0, 15.0,
        9.0, 14.0,
        9.0, 13.0,
        11.0, 15.0,
        11.0, 14.0,
        10.0, 14.0,
        10.0, 16.0,
        12.0, 13.0
    ]);
    let y: DVector<f64> = DVector::from_row_slice(&[82.0, 79.0, 74.0, 83.0, 80.0, 81.0, 84.0, 81.0]);

    let data: RegressionData = RegressionData::from_reference_without_intercept(&x, &y);
    let data_with_intercept = RegressionData::from_reference(&x, &y);

    let lm = data.fit_lm();
    lm.summary();
    let lm2 = data_with_intercept.fit_lm();
    lm2.summary();
}
