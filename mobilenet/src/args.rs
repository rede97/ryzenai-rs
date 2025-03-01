use std::path::PathBuf;

use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// disable NPU
    #[arg(long)]
    pub no_npu: bool,

    #[arg(short, long, default_value = "data/val_images")]
    pub val_images: PathBuf,
}
