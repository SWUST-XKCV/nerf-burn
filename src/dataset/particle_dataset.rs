use std::{
    fs,
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use burn::data::dataset::Dataset;
use image::{DynamicImage, ImageReader};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct ParticleItem {
    pub feature: [f32; 6], // shape: [6,], (x, y, z, x_dir, y_dir, z_dir)
    pub label: [i32; 3],   // shape: [3,], (r, g, b)
}

pub struct ParticleDataset {
    items: Vec<ParticleItem>,
}

impl Dataset<ParticleItem> for ParticleDataset {
    fn get(&self, index: usize) -> Option<ParticleItem> {
        if index > 0 && index < self.len() {
            Some(self.items[index].clone())
        } else {
            None
        }
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

impl ParticleDataset {
    pub fn train(root: impl AsRef<Path>) -> Self {
        Self::new(root, "train")
    }

    pub fn test(root: impl AsRef<Path>) -> Self {
        Self::new(root, "test")
    }

    pub fn valid(root: impl AsRef<Path>) -> Self {
        Self::new(root, "valid")
    }

    pub fn new(root: impl AsRef<Path>, split: impl AsRef<str>) -> Self {
        let file_names: [String; 2] = match split.as_ref() {
            "train" => ["train".to_string(), "transforms_train.json".to_string()],
            "valid" | "val" => ["val".to_string(), "transforms_val.json".to_string()],
            "test" => ["test".to_string(), "transforms_test.json".to_string()],
            _ => panic!("Invalid split specified {}", split.as_ref()),
        };

        let path_images_dir = root.as_ref().join(file_names[0].as_str());
        let path_transforms = root.as_ref().join(file_names[1].as_str());

        let (images, trans) =
            Self::read_data(path_images_dir, path_transforms).unwrap_or_else(|e| panic!("{:?}", e));

        let particles = Self::gen_sample_particles(images, trans);

        Self { items: particles }
    }

    fn gen_sample_particles(
        images: Mutex<Vec<DynamicImage>>,
        camera_trans: CameraTransformProcess,
    ) -> Vec<ParticleItem> {
        let images = images.lock().unwrap();
        let img_w = images[0].width();
        let img_h = images[0].height();
        let focal_len = 0.5 * img_w as f64 / f64::tan(camera_trans.camera_angle_x * 0.5);

        todo!()
    }

    fn gen_rays(w: usize, h: usize, focal_len: f64, c2w: [[f64; 4]; 4]) {
        todo!()
    }

    fn read_data(
        path_images_dir: impl AsRef<Path> + Sync + Send,
        path_transforms: impl AsRef<Path> + Sync + Send,
    ) -> Result<(Mutex<Vec<DynamicImage>>, CameraTransformProcess)> {
        let trans = Self::read_transforms(path_transforms)?;
        let n_img = trans.frames.len();
        let imgs = Mutex::new(vec![]);

        let range = 0..n_img;
        range.into_par_iter().for_each(|i| {
            let img = ImageReader::open(path_images_dir.as_ref().join(format!("r_{}.png", i)))
                .unwrap_or_else(|e| panic!("{:?}", e))
                .decode()
                .unwrap_or_else(|e| panic!("{:?}", e));
            imgs.lock().unwrap().push(img);
        });

        // [
        //  [-0.9,    0,   0,   0],
        //  [   0, -0.7, 0.6, 2.7],
        //  [ 0.0,  0.6, 0.7, 2.9],
        //  [ 0.0,  0.0, 0.0, 1.0]
        // ]

        Ok((imgs, trans))
    }

    fn read_transforms(path: impl AsRef<Path>) -> Result<CameraTransformProcess> {
        let text_json = fs::read_to_string(path.as_ref())?;
        let camera_trans: CameraTransformProcess = serde_json::from_str(text_json.as_ref())?;

        Ok(camera_trans)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CameraTransformProcess {
    camera_angle_x: f64,
    frames: Vec<Frame>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Frame {
    file_path: String,
    rotation: f64,
    transform_matrix: [[f64; 4]; 4],
}
