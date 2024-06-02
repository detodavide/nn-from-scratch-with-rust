extern crate ndarray;
mod nn_elements;

use ndarray::{Array1, array};
use nn_elements::layer::Layer;

fn main() {
    // Define an input vector of 3 dimensions
    let inputs: Array1<f64> = array![1.0, 2.0, 3.0];

    let layer: Layer = Layer::new(3, inputs.len());

    // Calculate the dot product and add the bias for each neuron
    // let neurons_no_bias: Array1<f64> = weights.dot(&inputs);
    // let neurons: Array1<f64> = neurons_no_bias + biases;

    // Print the results
    println!("Layer :\n{:?}", layer);
}
