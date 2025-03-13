mod camera_task;
mod cli_args;
mod image_task;
mod post_proc;

use ai_common::camera_sdl3;
use anyhow::Result;
use log::{info, warn};
use ort::execution_providers::{
    CPUExecutionProvider, DirectMLExecutionProvider, VitisAIExecutionProvider,
};
use ort::session::{Session, builder::GraphOptimizationLevel};

use clap::Parser;
use post_proc::*;

fn init_model(args: &cli_args::Args) -> Result<Session> {
    ort::init().with_name("resnet_cifar").commit()?;

    let mut providers = Vec::new();
    if let Ok(config_file) = ai_common::runtime::find_config_file("vaip_config.json", None) {
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
        .with_optimization_level(GraphOptimizationLevel::Disable)?
        .with_execution_providers(providers)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    info!("Load model: {}", model_path);

    return Ok(model);
}

fn main() -> Result<()> {
    let args = cli_args::Args::parse();

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

    // setup GPU backend before setup onnx runtime(dynamic runtime load)
    let post_proc = Box::new(PostProcCPU::new());

    match &args.command {
        cli_args::Command::Image { dir } => {
            image_task::images_task(&args, dir, post_proc)?;
        }
        cli_args::Command::Camera {
            list,
            dev_idx,
            fmt_idx,
        } => {
            if *list {
                camera_sdl3::print_list_all_cameras();
                return Ok(());
            } else {
                camera_task::camera_task(&args, post_proc, *dev_idx, *fmt_idx)?;
            }
        }
    }

    Ok(())
}
