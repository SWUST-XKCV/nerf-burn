use nerf_burn::model;

use burn::backend::wgpu;

fn main() {
    type MyBackend = wgpu::Wgpu<f32, i32>;
    let device = wgpu::WgpuDevice::default();

    let model_conf = model::NerfModelConfig::new(256, 63, 27, 4);
    let model: model::NerfModel<MyBackend> = model_conf.init(&device);
    println!("{}", model);
}
