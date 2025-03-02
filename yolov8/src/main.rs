mod args;
mod image;
mod post_prcess;

use std::path::{Path, PathBuf};

use ::image::Rgb;
use ai_common::measure_time;
use log::{error, info, warn};
use ndarray::prelude::*;
use ort::execution_providers::{
    CPUExecutionProvider, DirectMLExecutionProvider, VitisAIExecutionProvider,
};
use ort::inputs;
use ort::session::SessionOutputs;
use ort::session::{Session, builder::GraphOptimizationLevel};

use clap::Parser;
use image::*;
use post_prcess::*;

fn main() -> ort::Result<()> {
    let log_config = simplelog::ConfigBuilder::new()
        .set_time_level(log::LevelFilter::Trace)
        .build();
    simplelog::CombinedLogger::init(vec![simplelog::TermLogger::new(
        simplelog::LevelFilter::Info,
        log_config.clone(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    )])
    .unwrap();

    let args = args::Args::parse();

    let runtime_path = ai_common::runtime::init_runtime(None);
    info!("ONNX Runtime path: {:?}", runtime_path);

    ort::init().with_name("resnet_cifar").commit()?;

    let mut providers = Vec::new();
    if let Ok(config_file) = ai_common::runtime::find_config_file(runtime_path, "vaip_config.json")
    {
        info!("Config file: {:?}", config_file);
        if args.no_npu {
            warn!("NPU is disabled");
        } else {
            providers.push(
                VitisAIExecutionProvider::default()
                    .with_config_file(config_file.to_str().unwrap())
                    .with_cache_dir("./cache/")
                    .with_cache_key("modelcachekey")
                    .build()
                    .error_on_failure(),
            );
        }
    } else {
        warn!("Config file not found, VitisAIExecutionProvider will not be used");
    }
    providers.append(&mut vec![
        DirectMLExecutionProvider::default().build(),
        CPUExecutionProvider::default().build(),
    ]);

    let model_path = args.model.to_str().unwrap();

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers(providers)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    info!("Load model: {}", model_path);

    let font =
        ab_glyph::FontRef::try_from_slice(include_bytes!("../asserts/DejaVuSans.ttf")).unwrap();

    let output_path = PathBuf::from("output");
    if !output_path.exists() {
        std::fs::create_dir(&output_path).unwrap();
    }

    let images = ImageIterator::new("data", args.keep_aspect_ratio).unwrap();
    let (count, duration) = measure_time!({
        let mut images_count: usize = 0;
        for (i, image) in images.enumerate() {
            images_count += 1;
            info!("Image: {}: {:?}", i, image.path);
            let outputs: SessionOutputs<'_, '_> = model.run(inputs![image.array.view()]?)?;
            let mut img = image.img.to_rgb8();
            for batch_result in amd_yolov8m_post_prcess(outputs, 1, 0.5, 100, 0.7, 100)? {
                for box_info in batch_result {
                    info!("box info: {}", box_info);
                    let scaled_box = box_info.scale(image.scale);
                    println!(
                        "scaled_box: {:?}, {} {}",
                        scaled_box,
                        scaled_box.width(),
                        scaled_box.height()
                    );
                    // Draw rectangle and text on the image
                    let color = Rgb([255, 0, 0]); // Red color for the box
                    imageproc::drawing::draw_hollow_rect_mut(
                        &mut img,
                        imageproc::rect::Rect::at(scaled_box.x1 as i32, scaled_box.y1 as i32)
                            .of_size(30, 30),
                        color,
                    );

                    // // Draw label and score
                    // let text = format!("{}: {:.2}", box_info.label(), box_info.score);
                    // imageproc::drawing::draw_text_mut(
                    //     &mut img,
                    //     color,
                    //     scaled_box.x1 as i32,
                    //     (scaled_box.y1 as i32).saturating_sub(20), // Position text above the box
                    //     ab_glyph::PxScale::from(20.0),
                    //     &font,
                    //     &text,
                    // );

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
