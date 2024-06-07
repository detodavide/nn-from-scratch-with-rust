use ndarray::Array2;

#[derive(Debug)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
}

impl ActivationFunction {
    pub fn apply_array(&self, array: &Array2<f64>) -> Array2<f64> {
        match self {
            ActivationFunction::ReLU => array.mapv(ActivationFunction::relu),
            ActivationFunction::Sigmoid => array.mapv(ActivationFunction::sigmoid),
            ActivationFunction::Tanh => array.mapv(ActivationFunction::tanh),
        }
    }

    pub(crate) fn relu(x: f64) -> f64{
        x.max(0.0)
    }


    pub(crate) fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    pub(crate) fn tanh(x: f64) -> f64 {
        x.tanh()
    }
}