use burn::{data::dataloader::batcher::Batcher, prelude::*};

use super::ParticleItem;

#[derive(Clone)]
pub struct ParticleBatcher<B: Backend> {
    device: B::Device,
}

#[derive(Clone, Debug)]
pub struct ParticleBatch<B: Backend> {
    pub features: Tensor<B, 2>, // shape: [batch_size, 6], [batch_size, (x, y, z, x_dir, y_dir, z_dir)]
    pub labels: Tensor<B, 2>,   // shape: [batch_size, 3], [batch_size, (r, g, b)]
}

impl<B: Backend> ParticleBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

impl<B: Backend> Batcher<ParticleItem, ParticleBatch<B>> for ParticleBatcher<B> {
    /// Separate the features and the labels from samples
    /// item: shape[9,], (x, y, z, x_dir, y_dir, z_dir, r, g, b)
    /// feature: shape [6,], (x, y, z, x_dir, y_dir, z_dir)
    /// label: shape: [3,], (r, g, b)
    fn batch(&self, items: Vec<ParticleItem>) -> ParticleBatch<B> {
        let features: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| TensorData::from(item.feature))
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 9]))
            .collect();

        let labels: Vec<Tensor<B, 2>> = items
            .iter()
            .map(|item| TensorData::from(item.label))
            .map(|data| Tensor::<B, 1>::from_data(data, &self.device))
            .map(|tensor| tensor.reshape([1, 9]))
            .collect();

        let features = Tensor::cat(features, 0).to_device(&self.device);
        let labels = Tensor::cat(labels, 0).to_device(&self.device);

        ParticleBatch { features, labels }
    }
}
