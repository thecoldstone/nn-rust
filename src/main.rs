use dense_layer::LayerDense;
use ndarray::{Array1, Array2};
use plotters::{
    backend::BitMapBackend,
    chart::{ChartBuilder, LabelAreaPosition},
    drawing::IntoDrawingArea,
    element::Circle,
    style::{Color, BLUE, WHITE},
};

use crate::activation_functions::relu::ReLU;

mod activation_functions;
mod dense_layer;
mod spiral;

fn draw(x: Array2<f64>, y: Array1<u8>) -> Result<(), Box<dyn std::error::Error>> {
    let coordinates: Vec<(&f64, &f64)> = {
        let x_axis = x.slice(ndarray::s![.., 0]);
        let y_axis = x.slice(ndarray::s![.., 1]);
        x_axis.into_iter().zip(y_axis).collect()
    };

    let root = BitMapBackend::new("training.png", (600, 400)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut ctx = ChartBuilder::on(&root)
        .set_label_area_size(plotters::chart::LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .caption("Training Data", ("sans-serif", 40))
        .build_cartesian_2d(-1.0..1.0, -1.0..1.0)
        .unwrap();
    ctx.configure_mesh().draw().unwrap();
    ctx.draw_series(
        coordinates
            .iter()
            .map(|(x, y)| Circle::new((**x, **y), 2, BLUE.filled())),
    )
    .unwrap();

    root.present()?;
    Ok(())
}

fn main() {
    let (x, y) = spiral::create_data(100, 3);

    /* draw(x, y).unwrap() */
    let layer = LayerDense::new(2, 3);
    let mut relu = ReLU::new();

    let output = layer.forward(x.clone());
    relu.forward(output.clone());

    println!("{:?}", relu.output.unwrap().slice(ndarray::s![..5, ..]));
}
