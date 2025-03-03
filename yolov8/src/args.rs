use std::path::PathBuf;

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// disable NPU
    #[arg(long)]
    pub no_npu: bool,

    /// Path to the ONNX model
    #[arg(short, long, default_value = "models/yolov8m.onnx")]
    pub model: PathBuf,

    #[arg(short, long)]
    pub no_keep_aspect_ratio: bool,
}
