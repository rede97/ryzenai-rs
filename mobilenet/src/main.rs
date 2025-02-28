mod args;
mod image;

use log::{error, info, warn};
use ndarray::prelude::*;
use ort::execution_providers::{
    CPUExecutionProvider, DirectMLExecutionProvider, VitisAIExecutionProvider,
};
use ort::inputs;
use ort::session::{Session, builder::GraphOptimizationLevel};

use clap::Parser;

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

    let model = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers(providers)?
        .with_intra_threads(4)?
        .commit_from_file("models/mobilenetv2_int.onnx")?;

    let test_images = image::ImageIterator::new(&args.val_images);


    println!("inputs: {:?}", model.inputs);
    println!("outputs: {:?}", model.outputs);
    

    for (test_idx, test_img) in test_images.take(10).enumerate() {
        // let sub_test_data = test_data.slice(s![test_idx..test_idx + 1, .., .., ..]);
        // let sub_test_labels = test_labels.slice(s![test_idx..test_idx + 1, ..]);

        // let outputs = model.run(inputs!["input" => sub_test_data.view()]?)?;

        // for (output, label) in outputs[0]
        //     .try_extract_tensor::<f32>()?
        //     .into_owned()
        //     .axis_iter(Axis(0))
        //     .zip(sub_test_labels.axis_iter(Axis(0)))
        // {
        //     let expect_idx = label[0] as usize;
        //     let expect_label_name = LABEL_NAME[expect_idx];
        //     let predix_idx = output
        //         .iter()
        //         .enumerate()
        //         .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        //         .unwrap()
        //         .0;
        //     let predicted_label_name = LABEL_NAME[predix_idx];
        //     if expect_idx != predix_idx {
        //         error!(
        //             "test idx: {}, expect: {}:{}, predict: {}:{}",
        //             test_idx, expect_label_name, expect_idx, predicted_label_name, predix_idx
        //         );
        //         fail += 1;
        //     }
        // }
    }

    // info!(
    //     "Fail: {}/{} (LOSS: {}%)",
    //     fail,
    //     num_test,
    //     fail as f32 / num_test as f32 * 100.0
    // );

    Ok(())
}
