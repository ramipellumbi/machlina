use nalgebra::{DMatrix, DVector};

pub struct RegressionData {
    pub x: DMatrix<f64>,
    pub y: DVector<f64>,
}

impl RegressionData {
    pub fn from_reference_without_intercept(x: &DMatrix<f64>, y: &DVector<f64>) -> RegressionData {
        RegressionData { x: x.clone_owned(), y: y.clone_owned() }
    }

    pub fn from_data_without_reference(x: DMatrix<f64>, y: DVector<f64>) -> RegressionData {
        RegressionData { x, y }
    }

    pub fn from_reference(x: &DMatrix<f64>, y: &DVector<f64>) -> RegressionData {
        let new_x: DMatrix<f64> = x.clone_owned().insert_column(0, 1.0);
        RegressionData::from_reference_without_intercept(&new_x, y)
    }

    pub fn from_data(x: DMatrix<f64>, y: DVector<f64>) -> RegressionData {
        let new_x: DMatrix<f64> = x.insert_column(0, 1.0);
        RegressionData::from_data_without_reference(new_x, y)
    }
}