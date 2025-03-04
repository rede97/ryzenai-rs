mod args;
#[allow(unused)]
mod camera;
mod image;
mod post_proc;
#[allow(unused)]
mod post_proc_gpu;

extern crate ffmpeg_next as ffmpeg;

use ai_common::measure_time;
use anyhow::{Result, anyhow};
use colored::Colorize;
use ffmpeg::format::context::Input;
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
use post_proc::*;

pub fn images_task<P: AsRef<Path>>(args: &args::Args, dir: P, model: Session) -> ort::Result<()> {
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
            for batch_results in amd_yolov8m_post_prcess(outputs, 1, 0.5, 100, 0.7, 100)? {
                for result in batch_results {
                    info!(
                        "class: {}, score: {:.2}",
                        result.label().cyan().bold(),
                        result.score
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

fn camera_task(args: &args::Args, model: Session, mut video_input: Input) -> Result<()> {
    let video_stream = video_input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .expect("No video stream found");

    let video_stream_index = video_stream.index();
    let context_decoder =
        ffmpeg::codec::Context::from_parameters(video_stream.parameters()).unwrap();
    let mut decoder = context_decoder.decoder().video().unwrap();
    println!(
        "decoder output format: {:?} {}x{}",
        decoder.format(),
        decoder.width(),
        decoder.height()
    );

    use sdl2::event::Event;
    use sdl2::keyboard::Keycode;

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    let _window = video_subsystem
        .window("Yolov8 on RyzenAI", 800, 600)
        .opengl() // this line DOES NOT enable opengl, but allows you to create/get an OpenGL context from your window.
        .build()
        .unwrap();

    // let mut canvas = window
    //     .into_canvas()
    //     .accelerated()
    //     // .present_vsync()
    //     .build()
    //     .map_err(|e| e.to_string())
    //     .map_err(|e| anyhow!("Create canvase error: {}", e))?;
    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|e| anyhow!("Create EventPump: {}", e))?;

    'mainloop: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Option::Some(Keycode::Escape),
                    ..
                } => break 'mainloop,

                _ => {}
            }
        }
    }
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

    match &args.command {
        args::Command::Image { dir } => {
            let model = init_model(&args)?;
            return images_task(&args, dir, model).map_err(|e| anyhow!(e));
        }
        args::Command::Camera { list, idx } => {
            camera::VideoDeviceIter::register_all();
            if *list {
                let iter = camera::VideoDeviceIter::new()?;
                for (i, dev) in iter.enumerate() {
                    println!("Camera [{}]: {}, {}", i, dev.desc, dev.name);
                }
            } else {
                if let Some(dev) = camera::VideoDeviceIter::new()?
                    .enumerate()
                    .filter(|(i, _d)| *i == *idx)
                    .map(|(_i, d)| d)
                    .next()
                {
                    let model = init_model(&args)?;
                    ffmpeg::init().unwrap();

                    let dshow = ffmpeg::device::input::video()
                        .filter(|v| v.name() == "dshow")
                        .next()
                        .expect("No input device [dshow]");

                    println!("device: {}, path: {}", dev.desc, dev.name);
                    let ictx = ffmpeg::format::open(&format!("video={}", dev.desc), &dshow)
                        .expect("Failed to open camera");

                    let video_input = ictx.input();

                    // camera_task(&args, model, video_input)?;
                } else {
                    println!("Camera[{}] not found", *idx);
                }
            }
            Ok(())
        }
    }
}
