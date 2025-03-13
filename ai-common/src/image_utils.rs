use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgb};
use ndarray::Array4;
use ndarray::ArrayView3;
use std::collections::VecDeque;
use std::fs;
use std::ops::Mul;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy)]
pub enum ImageScale {
    KeepAspectRatio { scale_ratio: f32, aspect_ratio: f32 },
    ScaleRatio { wdith_ratio: f32, height_ratio: f32 },
}

pub struct ImageIterator {
    paths: VecDeque<PathBuf>,
    keep_aspect_ratio: bool,
}

impl ImageIterator {
    pub fn new<P: AsRef<Path>>(dir_path: P, keep_aspect_ratio: bool) -> std::io::Result<Self> {
        let mut paths = VecDeque::new();
        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            if let Some(ext) = entry.path().extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    paths.push_back(entry.path());
                }
            }
        }
        Ok(Self {
            paths,
            keep_aspect_ratio,
        })
    }
}

pub struct ProcessedImage {
    pub path: PathBuf,
    pub img: DynamicImage,
}

impl Iterator for ImageIterator {
    type Item = ProcessedImage;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.paths.pop_front() {
            let img = image::open(&path).ok()?;
            return Some(ProcessedImage { path, img });
        }
        return None;
    }
}

#[allow(unused)]
pub fn save_ndarray_as_png(array: ArrayView3<f32>, path: &str) -> Result<(), image::ImageError> {
    let (height, width, _) = array.dim();
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (array[[y as usize, x as usize, 0]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let g = (array[[y as usize, x as usize, 1]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let b = (array[[y as usize, x as usize, 2]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    img.save(path)
}
