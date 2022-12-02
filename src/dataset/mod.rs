use std::ops::DivAssign;

use nalgebra::{DMatrix, DVector};
use simba::scalar::SubsetOf;

use crate::traits::number::Number;

pub struct Dataset<'a, T> where T: Number {
    pub features: &'a mut DMatrix<T>,
    pub labels: &'a mut DVector<T>,
}

impl<'a, T> Dataset<'a, T> where T: Number,  f64: SubsetOf<T> {
    /// Create a new Dataset 
    pub fn new(features: &'a mut DMatrix<T>, labels: &'a mut DVector<T>) -> Self {
        Dataset { features, labels }
    }

    /// Normalize the features and labels with z-score standardization in place
    pub fn normalize_z(&mut self) -> () {
        // population variance
        let sd_features = self.features.variance().sqrt();
        let mean_features = T::one().neg() * self.features.mean();

        self.features.add_scalar_mut(mean_features);
        self.features.div_assign(sd_features);
    }

    /// Normalize the features and labels with min-max standardization in place
    pub fn normalize_min_max(&mut self) -> () {
        let min_val: T = DMatrix::min(self.features);
        let max_val: T = DMatrix::max(self.features);

        self.features.add_scalar_mut(T::one().neg() * min_val);
        self.features.div_assign(max_val - min_val);
    }
}