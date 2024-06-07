use prettytable::{Table, row, cell};
use ndarray::Array2;

pub fn display_array2(array: &Array2<f64>) {
    let mut table = Table::new();
    
    for row_data in array.rows() {
        let table_row = row_data.iter().map(|&value| cell!(format!("{:.2}", value))).collect::<Vec<_>>();
        table.add_row(row!(table_row[0], table_row[1], table_row[2])); // Assuming 3 columns per row
    }
    
    table.printstd();
}