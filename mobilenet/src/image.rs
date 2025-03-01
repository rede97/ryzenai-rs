use image::{DynamicImage, GenericImageView};
use ndarray::Array4;

use std::fs;
use std::path::PathBuf;

pub struct ImageIterator {
    entries: fs::ReadDir,
}

impl ImageIterator {
    pub fn new(image_dir: &PathBuf) -> Self {
        let entries = fs::read_dir(image_dir).expect("Failed to read directory");
        ImageIterator { entries }
    }
}

impl Iterator for ImageIterator {
    type Item = Array4<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(entry) = self.entries.next() {
            if let Ok(entry) = entry {
                let image_path = entry.path();
                if let Ok(img) = image::open(&image_path) {
                    let img = img.resize_exact(224, 224, image::imageops::FilterType::Triangle);
                    return Some(image_to_array(img));
                }
            }
        }
        None
    }
}

fn image_to_array(img: DynamicImage) -> Array4<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array4::<f32>::zeros((1, 3, height as usize, width as usize));
    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }
    array
}
