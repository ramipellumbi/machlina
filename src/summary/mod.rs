pub mod linear_model;

use crate::traits::number::Number;

fn fmt_number<T: Number>(num: T, width: usize, precision: usize, exp_pad: usize) -> String {
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
