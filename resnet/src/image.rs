use cifar_ten::{Cifar10, CifarResult};
use image::{ImageBuffer, Rgb};
use ndarray::{Array, Array2, Array4};
use ndarray::{Array3, ArrayView3};
use std::error::Error;

struct BGR2RGBIterator<'a> {
    data: &'a [u8],
    idx: usize,
}

impl<'a> BGR2RGBIterator<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, idx: 0 }
    }
}

impl Iterator for BGR2RGBIterator<'_> {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.data.len() {
            return None;
        } else {
            let chan = self.idx / (32 * 32);
            let rest = self.idx % (32 * 32);
            let idx = match chan {
                0 => rest + 2 * (32 * 32),
                1 => rest + 1 * (32 * 32),
                2 => rest,
                _ => panic!("Invalid index"),
            };
            self.idx += 1;
            Some(self.data[idx])
        }
    }
}

pub fn to_ndarray<T: std::convert::From<u8>>(
    r: CifarResult,
) -> Result<(Array4<T>, Array2<T>, Array4<T>, Array2<T>), Box<dyn Error>> {
    let train_data: Array4<T> = Array::from_shape_vec((50_000, 3, 32, 32), r.0)?.mapv(|x| x.into());
    let train_labels: Array2<T> = Array::from_shape_vec((50_000, 10), r.1)?.mapv(|x| x.into());
    let test_data: Array4<T> = Array::from_shape_vec(
        (10_000, 3, 32, 32),
        BGR2RGBIterator::new(r.2.as_slice()).collect(),
    )?
    .mapv(|x| x.into());
    let test_labels: Array2<T> = Array::from_shape_vec((10_000, 1), r.3)?.mapv(|x| x.into());

    Ok((train_data, train_labels, test_data, test_labels))
}

fn save_ndarray_as_png(array: ArrayView3<f32>, path: &str) -> Result<(), image::ImageError> {
    let (height, width, _) = array.dim();
    let mut img = ImageBuffer::new(width as u32, height as u32);

    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let r = (array[[y as usize, x as usize, 0]] * 255.0) as u8;
        let g = (array[[y as usize, x as usize, 1]] * 255.0) as u8;
        let b = (array[[y as usize, x as usize, 2]] * 255.0) as u8;
        *pixel = Rgb([r, g, b]);
    }

    img.save(path)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use ndarray::Array3;
    // use tempfile::NamedTempFile;

    #[test]
    fn test_save_ndarray_as_png() {
        // Create a small 2x2 RGB image
        let array = Array3::from_shape_vec(
            (2, 2, 3),
            vec![
                1.0, 0.0, 0.0, // Red pixel
                0.0, 1.0, 0.0, // Green pixel
                0.0, 0.0, 1.0, // Blue pixel
                1.0, 1.0, 0.0, // Yellow pixel
            ],
        )
        .unwrap();

        // Create a temporary file to save the image
        let temp_path = "./test.png";

        // Save the image
        let result = save_ndarray_as_png(array.view(), temp_path);
        assert!(result.is_ok());

        // Optionally, you could load the saved image and verify its contents
    }
}
