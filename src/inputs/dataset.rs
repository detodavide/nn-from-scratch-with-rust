use ndarray::{stack, Array1, Array2, Axis};

#[derive(Debug)]
pub struct Dataset {
    pub inputs: Array2<f64>,
}

impl Dataset {
    pub fn new(inputs: Vec<Array1<f64>>) -> Self {
        let array_inputs = stack(Axis(0), &inputs.iter().map(|a| a.view()).collect::<Vec<_>>()).unwrap();

        Dataset { inputs: array_inputs }
    }
}