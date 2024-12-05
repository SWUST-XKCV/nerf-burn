use burn::{
    nn::{Linear, LinearConfig, Relu, Sigmoid},
    prelude::*,
};

#[derive(Module, Debug)]
pub struct NerfModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    linear4: Linear<B>,
    linear5: Linear<B>,
    linear6: Linear<B>,
    linear7: Linear<B>,
    linear8: Linear<B>,
    linear_view: Linear<B>,
    linear_alpha: Linear<B>,
    linear_rgb: Linear<B>,
    relu: Relu,
    sigmoid: Sigmoid,
}

impl<B: Backend> NerfModel<B> {
    /// # Shapes
    ///     - X: [batch_size, (d_pt_position + d_view_direction)], [batch_size, (pt_position, view_direction)]
    ///     - Output: [batch_size, 4], [batch_size, (r, g, b, alpha)]
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let n1 = input.dims()[0];
        let n2 = input.dims()[1];
        let input_pt = input.clone().slice([0..n1, 0..63]); // [batch_size, d_pt_position]
        let input_dir = input.slice([0..n1, 63..n2]); // [batch_size, d_view_direction]

        let x = self.linear1.forward(input_pt.clone());
        let x = self.relu.forward(x);
        let x = self.linear2.forward(x);
        let x = self.relu.forward(x);
        let x = self.linear3.forward(x);
        let x = self.relu.forward(x);
        let x = self.linear4.forward(x);
        let x = self.relu.forward(x);
        let x = self.linear5.forward(Tensor::cat(vec![x, input_pt], 1));
        let x = self.relu.forward(x);
        let x = self.linear6.forward(x);
        let x = self.relu.forward(x);
        let x = self.linear7.forward(x);
        let x = self.relu.forward(x);
        let x = self.linear8.forward(x);
        let x = self.relu.forward(x); // optional
        let alpha = self.linear_alpha.forward(x.clone());
        let x = self.linear_view.forward(Tensor::cat(vec![x, input_dir], 1));
        let x = self.relu.forward(x);
        let x = self.linear_rgb.forward(x);
        let rgb = self.sigmoid.forward(x);

        Tensor::cat(vec![rgb, alpha], 1)
    }
}

#[derive(Config, Debug)]
pub struct NerfModelConfig {
    width_linear: usize,
    d_input_pt: usize,
    d_input_dir: usize,
    d_output: usize,
}

impl NerfModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> NerfModel<B> {
        NerfModel {
            linear1: LinearConfig::new(self.d_input_pt, self.width_linear).init(device),
            linear2: LinearConfig::new(self.width_linear, self.width_linear).init(device),
            linear3: LinearConfig::new(self.width_linear, self.width_linear).init(device),
            linear4: LinearConfig::new(self.width_linear, self.width_linear).init(device),
            linear5: LinearConfig::new(self.width_linear + self.d_input_pt, self.width_linear)
                .init(device),
            linear6: LinearConfig::new(self.width_linear, self.width_linear).init(device),
            linear7: LinearConfig::new(self.width_linear, self.width_linear).init(device),
            linear8: LinearConfig::new(self.width_linear, self.width_linear).init(device),
            linear_alpha: LinearConfig::new(self.width_linear, 1).init(device),
            linear_view: LinearConfig::new(
                self.width_linear + self.d_input_dir,
                self.width_linear / 2,
            )
            .init(device),
            linear_rgb: LinearConfig::new(self.width_linear / 2, 3).init(device),
            relu: Relu::new(),
            sigmoid: Sigmoid::new(),
        }
    }
}
