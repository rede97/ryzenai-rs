use cifar_ten::CifarResult;
use image::{ImageBuffer, Rgb};
use ndarray::ArrayView3;
use ndarray::{Array, Array2, Array4};
use std::error::Error;
use std::ops::Mul;

pub fn to_ndarray<T: std::convert::From<u8>>(
    r: CifarResult,
) -> Result<(Array4<T>, Array2<u8>, Array4<T>, Array2<u8>), Box<dyn Error>> {
    let train_data: Array4<T> = Array::from_shape_vec((50_000, 3, 32, 32), r.0)?.mapv(|x| x.into());
    let train_labels: Array2<u8> = Array::from_shape_vec((50_000, 1), r.1)?;
    let test_data: Array4<T> = Array::from_shape_vec((10_000, 3, 32, 32), r.2)?.mapv(|x| x.into());
    let test_labels: Array2<u8> = Array::from_shape_vec((10_000, 1), r.3)?;

    Ok((train_data, train_labels, test_data, test_labels))
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
