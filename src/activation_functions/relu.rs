use ndarray::Array2;

#[derive(Debug)]
pub struct ReLU {
    pub output: Option<Array2<f64>>,
}

impl ReLU {
    /// Creates ReLU activation function instance
    pub fn new() -> Self {
        Self { output: None }
    }

    /// Implementation of element-wise maximum comparison for one array
    pub fn forward(&mut self, inputs: Array2<f64>) {
        self.output = Some(inputs.map(|&x| {
            if x > 0.0 {
                x.to_owned()
            } else {
                0.0.to_owned()
            }
        }));
    }
}
