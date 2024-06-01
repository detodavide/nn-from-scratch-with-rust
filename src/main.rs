extern crate ndarray;
mod nn_elements;

use ndarray::{Array1, Array2, array};
use nn_elements::neuron::Neuron;

fn main() {
    // Define an input vector of 3 dimensions
    let inputs: Array1<f64> = array![1.0, 2.0, 3.0];

    let neuron: Neuron = Neuron::new(3);

    // Weights matrix to represent 3 neurons and 3 inputs -> 3x3
    let weights: Array2<f64> = array![
        [0.2, 0.8, 1.0],  //  each row represent a single neuron weights
        [0.3, -0.67, 0.4], 
        [1.0, 0.34, -0.4]
    ];

    // Biases for each neuron
    let biases: Array1<f64> = array![0.8, 2.0, 3.0];

    // Calculate the dot product and add the bias for each neuron
    let neurons_no_bias: Array1<f64> = weights.dot(&inputs);
    let neurons: Array1<f64> = neurons_no_bias + biases;

    // Print the results
    println!("Weights matrix:\n{:?}", weights);
    println!("Output of neurons: {:?}", neurons);
}
