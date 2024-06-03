extern crate ndarray;
mod nn_elements;
mod inputs;

use ndarray::{Array1, array, Array2};
use nn_elements::layer::Layer;
use inputs::dataset::Dataset;

fn main() {
    // Define an input vector of 3 dimensions
    let inputs: Vec<Array1<f64>> = vec![array![1.0, 2.0, 3.0],array![4.0, 32.0, 21.0]];
    let dataset: Dataset = Dataset::new(inputs.clone());
    println!("{:?}", dataset);

    let layer: Layer = Layer::new(3, 3);
    println!("{:?}", layer);

    // Perform matrix-vector multiplication 3x3 * 3x2 = 3x2, in the output the rows are the neurons and the columns are the batches
    let out: Array2<f64> = layer.matrix.dot(&dataset.inputs.t());
    println!("Output: {:?}", out);
}
