use plotters::{
    backend::BitMapBackend,
    chart::{ChartBuilder, LabelAreaPosition},
    drawing::IntoDrawingArea,
    element::Circle,
    style::{Color, BLUE, WHITE},
};

mod spiral;

pub trait NeuronTrait {
    fn mutate(&self, weights: &[f64], bias: f64) -> f64;
}

pub struct Neuron {
    pub input: [f64; 4],
}

impl NeuronTrait for Neuron {
    fn mutate(&self, weights: &[f64], bias: f64) -> f64 {
        let mut output = 0.0;

        for input in self.input.into_iter().enumerate() {
            let (i, x): (usize, f64) = input;
            output += x as f64 * weights[i]
        }

        return output + bias;
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (x, y) = spiral::create_data(100, 3);

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
