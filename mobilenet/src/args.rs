use std::path::PathBuf;

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// disable NPU
    #[arg(long)]
    pub no_npu: bool,

    /// Path to the test images
    #[arg(short, long, default_value = "data/val_images")]
    pub val_images: PathBuf,

    /// Path to the ONNX model
    #[arg(short, long, default_value = "models/resnet_quantized.onnx")]
    pub model: PathBuf,
}
