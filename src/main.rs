extern crate nalgebra as na;

use machlina::{dataset::Dataset, linear_model::linear_model::LinearModel, summary::Summary};
use na::{DMatrix, DVector};

fn main() {
    let mut x: DMatrix<f64> = DMatrix::from_row_slice(8, 2, &[
        10.0, 15.0,
        9.0, 14.0,
        9.0, 13.0,
        11.0, 15.0,
        11.0, 14.0,
        10.0, 14.0,
        10.0, 16.0,
        12.0, 13.0
    ]);
    let mut y: DVector<f64> = DVector::from_row_slice(&[82.0, 79.0, 74.0, 83.0, 80.0, 81.0, 84.0, 81.0]);

    let dataset = Dataset::new(&mut x, &mut y);

    let lm = dataset.fit_lm();
    lm.summary();


    let mut new_x = x.insert_column(0, 1.0);
    let mut new_y = y.clone_owned();
    let intercept_dataset = Dataset::new(&mut new_x, &mut new_y);
    let lm_intercept = intercept_dataset.fit_lm();
    lm_intercept.summary();

}
