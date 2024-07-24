use ndarray::Array2;
use ndarray_rand::{rand_distr::Normal, RandomExt};

#[derive(Debug)]
pub struct Neuron {
    pub inputs: usize,
    pub neurons: usize,
}

#[derive(Debug)]
pub struct LayerDense {
    pub weights: Array2<f64>,
    pub biases: Array2<f64>,
    pub neuron: Neuron,
}

impl LayerDense {
    pub fn new(inputs: usize, neurons: usize) -> Self {
        Self {
            neuron: Neuron { inputs, neurons },
            weights: Array2::random((inputs, neurons), Normal::new(0.0, 1.0).unwrap()) * 0.01,
            biases: Array2::<f64>::zeros((1, neurons)),
        }
    }

    pub fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        inputs.dot(&self.weights) + &self.biases
    }
}
