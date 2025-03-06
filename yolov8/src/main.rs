mod args;
mod image;
mod post_proc;

use ai_common::measure_time;
use anyhow::{Result, anyhow};
use colored::Colorize;
use log::{info, warn};
use ort::execution_providers::{
    CPUExecutionProvider, DirectMLExecutionProvider, VitisAIExecutionProvider,
};
use ort::inputs;
use ort::session::SessionOutputs;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::{Path, PathBuf};

use clap::Parser;
use image::*;
#[allow(unused)]
use post_proc::*;

pub fn images_task<P: AsRef<Path>>(
    args: &args::Args,
    dir: P,
    model: Session,
    proc: Box<dyn AMDYoloV8PostProc>,
) -> Result<()> {
    let font =
        ab_glyph::FontRef::try_from_slice(include_bytes!("../asserts/DejaVuSans.ttf")).unwrap();
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
                    let color = result.color(); // Red color for the box
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

fn camera_task(_args: &args::Args, _model: Session) -> Result<()> {
    Ok(())
}

fn init_model(args: &args::Args) -> Result<Session> {
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

    return Ok(model);
}

fn main() -> Result<()> {
    let args = args::Args::parse();

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

    /// setup GPU backend before setup onnx runtime
    let post_proc: Box<dyn AMDYoloV8PostProc> = if args.gpu_post {
        Box::new(PostProcWGPU::new()?)
    } else {
        Box::new(PostProcCPU::new())
    };

    match &args.command {
        args::Command::Image { dir } => {
            let model = init_model(&args)?;
            return images_task(&args, dir, model, post_proc).map_err(|e| anyhow!(e));
        }
        args::Command::Camera { list, idx: _ } => {
            if *list {
                todo!()
            } else {
                let model = init_model(&args)?;
                camera_task(&args, model)?;
            }
            Ok(())
        }
    }
}
