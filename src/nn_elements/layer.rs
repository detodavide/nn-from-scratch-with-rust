use ndarray::Array2;
use crate::nn_elements::neuron::Neuron;
use crate::nn_elements::activation_func::ActivationFunction;

#[derive(Debug)]
pub struct DenseLayer {
    pub neurons: Vec<Neuron>,
    pub weights_matrix: Array2<f64>,
    pub biases: Array2<f64>,
    pub activation: ActivationFunction
}

impl DenseLayer {
    pub fn new(
        n_neurons: usize,
        n_inputs: usize, // This value represent the number of inputs to the layer
        activation: ActivationFunction
    ) -> Self {
        let mut neurons = Vec::with_capacity(n_neurons);
        let mut weights = Vec::with_capacity(n_neurons * n_inputs);
        let mut biases = Vec::with_capacity(n_neurons); // Prepare a vector to hold neuron biases

        for _ in 0..n_neurons {
            let neuron = Neuron::new(n_inputs);
            weights.extend(neuron.weights.iter().cloned());
            neurons.push(neuron.clone());
            biases.push(neuron.bias);
        }

        let weights_matrix = Array2::from_shape_vec((n_neurons, n_inputs), weights).unwrap();
        let biases_array = Array2::from_shape_vec((n_neurons, 1), biases).unwrap();

        DenseLayer {
            neurons,
            weights_matrix,
            biases: biases_array,
            activation
        }
    }

    pub fn forward(&self, inputs: Array2<f64>) -> Array2<f64> {
        let mut output = self.weights_matrix.dot(&inputs.t());
        output += &self.biases;
        let activated_output = self.activation.apply_array(&output);
        activated_output.t().to_owned()
    }
}
