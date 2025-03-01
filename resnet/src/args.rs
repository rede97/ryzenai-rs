use clap::Parser;

/// Simple program to greet a person
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    /// disable NPU
    #[arg(long)]
    pub no_npu: bool,
}
