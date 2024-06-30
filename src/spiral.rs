use std::{ops::Mul, usize};

use ndarray::{stack, Array, Array1, Array2, Axis, Ix1};
use ndarray_rand::{rand_distr::Normal, RandomExt};

pub fn sin(mut matrix: Array<f64, Ix1>) -> Array<f64, Ix1> {
    matrix.mapv_into(|v| v.exp().sin())
}

pub fn cos(mut matrix: Array<f64, Ix1>) -> Array<f64, Ix1> {
    matrix.mapv_into(|v| v.exp().cos())
}

pub fn create_data(samples: usize, classes: usize) -> (Array2<f64>, Array1<u8>) {
    let mut x = Array2::<f64>::zeros((samples.mul(classes) as usize, 2));
    let mut y = Array1::<u8>::zeros(samples.mul(classes) as usize);

    for class_number in 0..classes {
        let r = Array::linspace(0., 1., samples as usize);
        let rand = Array::random(samples, Normal::new(0.0, 1.0).unwrap()).mul(0.2);
        let t = Array::linspace(
            class_number.mul(4) as f64,
            (class_number + 1).mul(4) as f64,
            samples as usize,
        ) + rand.clone();
        y.slice_mut(ndarray::s![class_number * samples..(class_number + 1) * samples;1])
            .fill(class_number as u8);

        let mut slice = x.slice_mut(ndarray::s![
            class_number * samples..(class_number + 1) * samples,
            ..
        ]);

        slice.assign(&stack![
            Axis(1),
            r.clone() * sin(t.clone() * 2.5),
            r * cos(t * 2.5)
        ]);
    }

    (x, y)
}
