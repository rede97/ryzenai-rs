use crate::{cli_args, post_proc::*};
use ai_common::image_utils::ImageIterator;
use ai_common::measure_time;
use anyhow::Result;
use colored::Colorize;
use log::info;
use ort::inputs;
use ort::session::SessionOutputs;
use std::path::{Path, PathBuf};

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

    let images = ImageIterator::new(dir, !args.no_keep_aspect_ratio).unwrap();
    let (count, duration) = measure_time!({
        let mut images_count: usize = 0;
        for (i, image) in images.enumerate() {
            images_count += 1;
            info!("Image {}: {:?}", i, image.path);
            let outputs: SessionOutputs<'_, '_> = model.run(inputs![image.array.view()]?)?;
            let mut img = image.img.to_rgb8();
            let (results, duration) = measure_time!({
                let results = proc.post_proc(outputs, 1, 0.5, 100, 0.7, 100)?;
                results
            });
            info!("post proc time: {}us", duration.as_micros());
            for batch_results in results {
                for result in batch_results {
                    info!(
                        "class: {}, score: {:.2} bbox: {:?}",
                        result.label().cyan().bold(),
                        result.score,
                        result.bbox,
                    );
                    let scaled_box: BoundingBox = result.scale(image.scale);
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
