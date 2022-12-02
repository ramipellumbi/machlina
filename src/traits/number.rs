use nalgebra::{Field, RealField, Scalar, SimdPartialOrd, SimdValue};

use num_traits::{ FromPrimitive, Signed, AsPrimitive};
use std::fmt::LowerExp;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

pub trait Number:
    Field
    + Copy
    + AsPrimitive<f64>
    + RealField
    + FromPrimitive
    + Signed
    + Sum
    + SimdValue
    + SimdPartialOrd
    + LowerExp
    + for<'a> AddAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + std::marker::Unpin
    + Scalar
    {}

impl Number for f64 {}

impl Number for f32 {}