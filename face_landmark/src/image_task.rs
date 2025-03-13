use crate::{cli_args, post_proc::*};
use ai_common::image_utils::{ImageIterator, ImageScale};
use ai_common::measure_time;
use anyhow::Result;
use colored::Colorize;
use image::imageops::FilterType;
use image::{DynamicImage, ImageBuffer, Rgb};
use log::info;
use ndarray::Array4;
use ort::inputs;
use ort::session::SessionOutputs;
use std::path::{Path, PathBuf};

fn image_convert(img: &DynamicImage) -> (ndarray::Array4<f32>, ImageScale) {
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
    } else if width != height {
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

    let mut array = Array4::<f32>::zeros((1, 3, 640, 640));
    for y in 0..640 {
        for x in 0..640 {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            array[[0, 0, y, x]] = pixel[0] as f32 / 255.0;
            array[[0, 1, y, x]] = pixel[1] as f32 / 255.0;
            array[[0, 2, y, x]] = pixel[2] as f32 / 255.0;
        }
    }
    return (array, scale);
}

pub fn images_task<P: AsRef<Path>>(
    args: &cli_args::Args,
    dir: P,
    proc: Box<dyn AMDYoloV8PostProc>,
) -> Result<()> {
    let model = crate::init_model(&args)?;
    let font = ab_glyph::FontRef::try_from_slice(ai_common::ttf_sdl3::FONT_DATA).unwrap();
    let output_path = PathBuf::from("output");
    if !output_path.exists() {
        std::fs::create_dir(&output_path).unwrap();
    }

    let images = ImageIterator::new(dir).unwrap();
    let (count, duration) = measure_time!({
        let mut images_count: usize = 0;
        for (i, image) in images.enumerate() {
            images_count += 1;
            info!("Image {}: {:?}", i, image.path);
            let (img_array, img_scale) = image_convert(&image.img);
            let (results, duration) = measure_time!({
                let outputs: SessionOutputs<'_, '_> = model.run(inputs![img_array.view()]?)?;
                let results = proc.post_proc(outputs, 1, 0.5, 100, 0.7, 100)?;
                results
            });
            info!("infra time: {}ms", duration.as_micros() as f32 / 100.0);
            let mut img = image.img.to_rgb8();
            for batch_results in results {
                for result in batch_results {
                    info!(
                        "class: {}, score: {:.2} bbox: {:?}",
                        result.label().cyan().bold(),
                        result.score,
                        result.bbox,
                    );
                    let scaled_box: BoundingBox = result.scale(img_scale);
                    // Draw rectangle and text on the image
                    let color = result.img_color(); // Red color for the box
                    imageproc::drawing::draw_hollow_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(scaled_box.x1 as i32, scaled_box.y1 as i32)
                            .of_size(scaled_box.width() as u32, scaled_box.height() as u32),
                        color,
                    );

                    // Draw label and score
                    let text = format!("{}: {:.2}", result.label(), result.score);
                    imageproc::drawing::draw_text_mut(
                        &mut img,
                        color,
                        scaled_box.x1 as i32,
                        (scaled_box.y1 as i32).saturating_sub(20), // Position text above the box
                        ab_glyph::PxScale::from(20.0),
                        &font,
                        &text,
                    );

                    // Save the annotated image
                }
            }
            let output_name = image.path.file_name().unwrap().to_str().unwrap();
            img.save(output_path.join(output_name)).unwrap();
        }
        images_count
    });

    info!(
        "Duration: {:?}, FPS: {:.2}",
        duration,
        count as f64 / duration.as_secs_f64()
    );

    Ok(())
}
