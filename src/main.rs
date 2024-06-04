extern crate ndarray;
mod nn_elements;
mod inputs;

use ndarray::{Array1, array, Array2};
use nn_elements::layer::DenseLayer;
use inputs::dataset::Dataset;

fn main() {
    // Define an input vector of 3 dimensions
    let inputs: Vec<Array1<f64>> = vec![array![1.0, 2.0, 3.0],array![4.0, 32.0, 21.0]];
    let dataset: Dataset = Dataset::new(inputs.clone());
    println!("{:?}", dataset);

    let layer1: DenseLayer = DenseLayer::new(3, 3);
    let layer2: DenseLayer = DenseLayer::new(4, 3);
    println!("{:?}", layer1);

    // Perform matrix-vector multiplication 3x3 * 3x2 = 3x2, in the output the rows are the neurons and the columns are the batches
    let out1: Array2<f64> = layer1.weights_matrix.dot(&dataset.inputs.t());
    let out1_with_bias: Array2<f64> = &out1 + layer1.biases;

    // weights matrix 4x3, first_layer_output 3x2, 4x3 * 3x2 = 4x2
    let out2: Array2<f64> = layer2.weights_matrix.dot(&out1_with_bias);
    let out2_with_bias: Array2<f64> = &out2 + layer2.biases;

    println!("Output no bias: {:?}", out2);
    println!("Output: {:?}", out2_with_bias);
}
