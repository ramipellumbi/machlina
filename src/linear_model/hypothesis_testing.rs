use crate::distributions::{pt, pf};

pub fn one_sided_t_test(t_val: f64, dof: usize) -> f64 {
    1.0 - pt(t_val.abs(), dof)
}

pub fn two_sided_t_test(t_val: f64, dof: usize) -> f64 {
    2.0 * one_sided_t_test(t_val, dof) 
}

pub fn one_sided_f_test(f_val: f64, ndof: usize, ddof: usize) -> f64 {
    1.0 - pf(f_val.abs(), ndof, ddof)
}

pub fn two_sided_f_test(f_val: f64, ndof: usize, ddof: usize) -> f64 {
    2.0 * one_sided_f_test(f_val, ndof, ddof) 
}