use ndarray::Array2;
use crate::nn_elements::neuron::Neuron;

#[derive(Debug)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
    pub matrix: Array2<f64>,
}

impl Layer {
    pub fn new(n_neurons: usize, n_weights: usize) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        let mut weights = Vec::with_capacity(n_neurons * n_weights);

        for _ in 0..n_neurons {
            let neuron = Neuron::new(n_weights);
            weights.extend(neuron.weights.iter().cloned());
            neurons.push(neuron);
        }

        let matrix = Array2::from_shape_vec((n_neurons, n_weights), weights).unwrap();

        Layer {
            neurons,
            matrix,
        }
    }
}
