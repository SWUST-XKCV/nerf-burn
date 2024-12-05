mod test {
    use std::{path::PathBuf, str::FromStr};

    use nerf_burn::dataset::ParticleDataset;

    #[test]
    fn lego_dataset_train() {
        let root = PathBuf::from_str("dataset/blender-synthesis-dataset/lego").unwrap();
        let dataset_train = ParticleDataset::train(root);
    }
}
