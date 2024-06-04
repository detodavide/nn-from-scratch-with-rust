use ndarray::Array1;
use rand::Rng;

#[derive(Clone)] 
#[derive(Debug)]
pub struct Neuron {
    pub weights: Array1<f64>,
    pub bias: f64

}

impl Neuron {
    // Method to create a new neuron with a given number of inputs
    pub fn new(num_inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        let weights = Array1::from_shape_fn(num_inputs, |_| rng.gen_range(-1.0..1.0));
        let bias = rng.gen_range(-1.0..1.0);

        Neuron { weights, bias }
    }
}