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

    /// Keep aspect ratio while sending image into model
    #[arg(short, long)]
    pub no_keep_aspect_ratio: bool,

    /// Use gpu for post processing
    #[arg(short, long)]
    pub gpu_post: bool,
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
        #[arg(short, long, default_value = "false")]
        list: bool,
        /// Camera device index
        #[arg(short, long)]
        dev_idx: Option<usize>,

        /// Camera format index
        #[arg(short, long)]
        fmt_idx: Option<usize>,
    },
}
