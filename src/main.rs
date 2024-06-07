extern crate ndarray;
mod nn_elements;
mod inputs;
mod utils;

use ndarray::{Array1, array, Array2};
use nn_elements::layer::DenseLayer;
use nn_elements::activation_func::ActivationFunction;
use inputs::dataset::Dataset;

fn main() {
    // Define an input vector of 3 dimensions
    let inputs: Vec<Array1<f64>> = vec![array![1.0, 2.0, 3.0],array![4.0, 32.0, 21.0]];
    let dataset: Dataset = Dataset::new(inputs.clone());
    println!("{:?}", dataset);

    let input_matrix: &Array2<f64> = dataset.get_inputs();
    
    let layer1: DenseLayer = DenseLayer::new(3, 3, ActivationFunction::ReLU);

    // Perform the forward pass
    let output1: Array2<f64> = layer1.forward(input_matrix.clone());

    println!(" First layer Output: {:?}", output1);

    let layer2: DenseLayer = DenseLayer::new(4, 3, ActivationFunction::ReLU);
    let output1: Array2<f64> = layer2.forward(output1.clone());

    println!("Final Output: {:?}", output1);
}
