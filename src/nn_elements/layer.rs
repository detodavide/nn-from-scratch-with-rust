use ndarray::Array2;
use crate::nn_elements::neuron::Neuron;

#[derive(Debug)]
pub struct DenseLayer {
    pub neurons: Vec<Neuron>,
    pub weights_matrix: Array2<f64>,
    pub biases: Array2<f64>,
}

impl DenseLayer {
    pub fn new(
        n_neurons: usize,
        n_weights: usize // This value represent the number of inputs to the layer
    ) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        let mut weights = Vec::with_capacity(n_neurons * n_weights);
        let mut biases = Vec::with_capacity(n_neurons); // Prepare a vector to hold neuron biases

        for _ in 0..n_neurons {
            let neuron = Neuron::new(n_weights);
            weights.extend(neuron.weights.iter().cloned());
            neurons.push(neuron.clone());
            biases.push(neuron.bias);
        }

        let weights_matrix = Array2::from_shape_vec((n_neurons, n_weights), weights).unwrap();
        let biases_array = Array2::from_shape_vec((n_neurons, 1), biases).unwrap(); // Convert biases vector to Array2

        DenseLayer {
            neurons,
            weights_matrix,
            biases: biases_array
        }
    }
}
