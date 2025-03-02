use image::imageops::FilterType;
use image::{ImageBuffer, Rgb};
use ndarray::Array4;
use ndarray::ArrayView3;
use std::collections::VecDeque;
use std::fs;
use std::ops::Mul;
use std::path::Path;

#[derive(Debug)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

pub struct ImageIterator {
    paths: VecDeque<String>,
    keep_aspect_ratio: bool,
}

impl ImageIterator {
    pub fn new<P: AsRef<Path>>(dir_path: P, keep_aspect_ratio: bool) -> std::io::Result<Self> {
        let mut paths = VecDeque::new();
        for entry in fs::read_dir(dir_path)? {
            let entry = entry?;
            if let Some(ext) = entry.path().extension() {
                if ext == "jpg" || ext == "jpeg" || ext == "png" {
                    paths.push_back(entry.path().to_string_lossy().into_owned());
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
    pub path: String,
    pub array: Array4<f32>,
    pub original_size: ImageSize,
}

impl Iterator for ImageIterator {
    type Item = ProcessedImage;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.paths.pop_front() {
            let img = image::open(&path).ok()?;
            let original_size = ImageSize {
                width: img.width(),
                height: img.height(),
            };

            // Skip resizing if image is already 640x640
            let rgb = if original_size.width == 640 && original_size.height == 640 {
                img.to_rgb8()
            } else if self.keep_aspect_ratio && original_size.width != original_size.height {
                // 计算缩放比例
                let width_ratio = 640.0 / original_size.width as f32;
                let height_ratio = 640.0 / original_size.height as f32;
                let ratio = width_ratio.min(height_ratio);

                // 计算新尺寸
                let new_width = (original_size.width as f32 * ratio).round() as u32;
                let new_height = (original_size.height as f32 * ratio).round() as u32;

                // 等比例缩放
                let resized = img.resize(new_width, new_height, FilterType::Triangle);

                // 创建640x640的黑色背景
                let mut canvas = ImageBuffer::new(640, 640);
                for pixel in canvas.pixels_mut() {
                    *pixel = Rgb([0, 0, 0]);
                }

                // 计算居中位置
                let x_offset = (640 - new_width) / 2;
                let y_offset = (640 - new_height) / 2;

                // 将缩放后的图像绘制到画布上
                image::imageops::overlay(
                    &mut canvas,
                    &resized.to_rgb8(),
                    x_offset as i64,
                    y_offset as i64,
                );
                canvas
            } else {
                // 直接缩放
                img.resize_exact(640, 640, FilterType::Triangle).to_rgb8()
            };

            let mut array = Array4::<f32>::zeros((1, 640, 640, 3));
            for y in 0..640 {
                for x in 0..640 {
                    let pixel = rgb.get_pixel(x as u32, y as u32);
                    array[[0, y, x, 0]] = pixel[0] as f32 / 255.0;
                    array[[0, y, x, 1]] = pixel[1] as f32 / 255.0;
                    array[[0, y, x, 2]] = pixel[2] as f32 / 255.0;
                }
            }
            Some(ProcessedImage {
                path,
                array,
                original_size,
            })
        } else {
            None
        }
    }
}

#[allow(unused)]
pub fn save_ndarray_as_png(array: ArrayView3<f32>, path: &str) -> Result<(), image::ImageError> {
    let (_, height, width) = array.dim();
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (array[[0, y as usize, x as usize]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let g = (array[[1, y as usize, x as usize]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        let b = (array[[2, y as usize, x as usize]]
            .mul(255.0)
            .clamp(0., 255.)) as u8;
        *pixel = Rgb([r, g, b]);
    }

    img.save(path)
}
