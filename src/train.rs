use burn::{
    data::dataloader::DataLoaderBuilder, optim::AdamConfig, prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::Rng;

use crate::{
    dataset::{ParticleBatcher, ParticleDataset},
    model::{NerfModel, NerfModelConfig},
};

#[derive(Config)]
pub struct TrainingConfig {
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 1024)]
    pub batch_size: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 3e-4)]
    pub lr: f64,
    #[config(default = 4)]
    pub num_workers: usize,
    pub model: NerfModelConfig,
    pub optimizer: AdamConfig,
}

pub fn train<B: AutodiffBackend>(device: &B::Device) {
    let config_model = NerfModelConfig::new(256, 63, 27, 4);
    let config_optimizer = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.9999)
        .with_epsilon(1e-10);
    let config =
        TrainingConfig::new(config_model, config_optimizer).with_seed(rand::thread_rng().gen());

    B::seed(config.seed);

    let mut model: NerfModel<B> = config.model.init(device);
    //let mut optim = config.optimizer.init();

    let batcher_train = ParticleBatcher::<B>::new(device.clone());
    let batcher_test = ParticleBatcher::<B::InnerBackend>::new(device.clone());
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ParticleDataset::train(
            "dataset/blender-synthesis-dataset/lego/",
        ));
    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(ParticleDataset::test(
            "dataset/blender-synthesis-dataset/lego/",
        ));
    todo!();
}
