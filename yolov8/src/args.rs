use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[command(subcommand)]
    pub command: Command,

    /// disable NPU
    #[arg(long)]
    pub no_npu: bool,

    /// Path to the ONNX model
    #[arg(short, long, default_value = "models/yolov8m.onnx")]
    pub model: PathBuf,

    #[arg(short, long)]
    pub no_keep_aspect_ratio: bool,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    #[command(name = "img", about = "Image mode")]
    Image {
        /// Input image dir path
        #[arg(short, long, default_value = "./data")]
        dir: PathBuf,
    },

    #[command(name = "cam", about = "Camera mode")]
    Camera {
        /// List all camera device
        #[arg(long, default_value = "false")]
        list: bool,
        /// Camera index
        #[arg(long, default_value = "0")]
        idx: usize,
    },
}
