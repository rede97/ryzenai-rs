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
    pub array: Array4<f32>,
    pub scale: ImageScale,
}

impl Iterator for ImageIterator {
    type Item = ProcessedImage;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(path) = self.paths.pop_front() {
            let img = image::open(&path).ok()?;

            let width = img.width();
            let height = img.height();

            // Skip resizing if image is already 640x640
            let (rgb, scale) = if width == 640 && height == 640 {
                (
                    img.to_rgb8(),
                    ImageScale::KeepAspectRatio {
                        scale_ratio: 1.0,
                        aspect_ratio: 1.0,
                    },
                )
            } else if self.keep_aspect_ratio && width != height {
                let width_ratio = 640.0 / width as f32;
                let height_ratio = 640.0 / height as f32;
                let scale_ratio = width_ratio.min(height_ratio);

                let new_width = (width as f32 * scale_ratio).round() as u32;
                let new_height = (height as f32 * scale_ratio).round() as u32;

                let resized = img.resize(new_width, new_height, FilterType::Triangle);

                let mut canvas = ImageBuffer::new(640, 640);
                for pixel in canvas.pixels_mut() {
                    *pixel = Rgb([0, 0, 0]);
                }

                let x_offset = (640 - new_width) / 2;
                let y_offset = (640 - new_height) / 2;

                image::imageops::overlay(
                    &mut canvas,
                    &resized.to_rgb8(),
                    x_offset as i64,
                    y_offset as i64,
                );
                (
                    canvas,
                    ImageScale::KeepAspectRatio {
                        scale_ratio,
                        aspect_ratio: width as f32 / height as f32,
                    },
                )
            } else {
                (
                    img.resize_exact(640, 640, FilterType::Triangle).to_rgb8(),
                    ImageScale::ScaleRatio {
                        wdith_ratio: 640.0 / img.width() as f32,
                        height_ratio: 640.0 / img.height() as f32,
                    },
                )
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
                img,
                array,
                scale,
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
