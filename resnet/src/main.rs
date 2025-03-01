mod args;
mod image;

use ai_common::measure_time;
use cifar_ten::Cifar10;
use log::{error, info, warn};
use ndarray::prelude::*;
use ort::execution_providers::{
    CPUExecutionProvider, DirectMLExecutionProvider, VitisAIExecutionProvider,
};
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};

use clap::Parser;
use image::*;

const LABEL_NAME: [&'static str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

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

    let (_train_data, _train_labels, test_data, test_labels) = to_ndarray::<f32>(
        Cifar10::default()
            // .download_and_extract(true)
            // .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
            .encode_one_hot(false)
            .build()
            .unwrap(),
    )
    .expect("Failed to build CIFAR-10 data");
    let test_data = test_data.mapv(|x| x / 255.0);
    info!("Load CIFAR-10 data done");

    let num_test = 1000;
    // let num_test = test_data.len_of(Axis(0));
    let mut fail = 0;

    let (_, duration) = measure_time!({
        for test_idx in 0..num_test {
            let sub_test_data = test_data.slice(s![test_idx..test_idx + 1, .., .., ..]);
            let sub_test_labels = test_labels.slice(s![test_idx..test_idx + 1, ..]);

            let outputs = model.run(inputs![sub_test_data.view()]?)?;

            for (output, label) in outputs[0]
                .try_extract_tensor::<f32>()?
                .into_owned()
                .axis_iter(Axis(0))
                .zip(sub_test_labels.axis_iter(Axis(0)))
            {
                let expect_idx = label[0] as usize;
                let expect_label_name = LABEL_NAME[expect_idx];
                let predix_idx = output
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .unwrap()
                    .0;
                let predicted_label_name = LABEL_NAME[predix_idx];
                if expect_idx != predix_idx {
                    error!(
                        "test idx: {}, expect: {}:{}, predict: {}:{}",
                        test_idx, expect_label_name, expect_idx, predicted_label_name, predix_idx
                    );
                    fail += 1;
                }
            }
        }
    });

    info!(
        "Fail: {}/{} (LOSS: {}%)",
        fail,
        num_test,
        fail as f32 / num_test as f32 * 100.0
    );
    info!(
        "Duration: {:?}, FPS: {}",
        duration,
        num_test as f32 / duration.as_secs_f32()
    );

    Ok(())
}
