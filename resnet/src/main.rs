mod image;

use std::error::Error;
use std::iter::zip;
use std::os;

use ort::execution_providers::{DirectMLExecutionProvider, VitisAIExecutionProvider};
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
// use ort::execution_providers::VitisAIExecutionProvider;
use cifar_ten::Cifar10;
use ndarray::prelude::*;
use ort::tensor::ArrayExtensions;
use windows::core::PCSTR;
use windows::Win32::System::LibraryLoader::SetDllDirectoryA;

use image::*;

fn main() -> ort::Result<()> {
    println!("Hello, world!");

    unsafe {
        let path = PCSTR::from_raw(b"../runtime/\0".as_ptr());
        if SetDllDirectoryA(path).is_ok() {
            println!("DLL search path set successfully.");
        } else {
            println!("Failed to set DLL search path.");
        }
    }

    let (_train_data, _train_labels, test_data, test_labels) = to_ndarray::<f32>(
        Cifar10::default()
            // .download_and_extract(true)
            // .download_url("https://cmoran.xyz/data/cifar/cifar-10-binary.tar.gz")
            .encode_one_hot(false)
            .build()
            .unwrap(),
    )
    .expect("Failed to build CIFAR-10 data");
    println!("Load CIFAR-10 data done");

    ort::init().with_name("resnet_cifar").commit()?;

    let model = Session::builder()?
        // .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_execution_providers([
            VitisAIExecutionProvider::default()
                .with_config_file("../runtime/vaip_config.json")
                .with_cache_dir("./cache/")
                .with_cache_key("modelcachekey")
                .build()
                .error_on_failure(),
            // DirectMLExecutionProvider::default().build(),
        ])?
        .with_intra_threads(4)?
        .commit_from_file("models/resnet_quantized.onnx")?;

    for (idx, input) in model.inputs.iter().enumerate() {
        println!("model.inputs[{}] {}: {}", idx, input.name, input.input_type);
    }

    let sub_test_data = test_data.slice(s![0..1, .., .., ..]);
    println!("{:?}", sub_test_data.shape());

    let outputs = model.run(inputs!["input" => sub_test_data.view()]?)?;
    let output = outputs["output"]
        .try_extract_tensor::<f32>()?
        // .t()
        .into_owned();
    for (o, label) in output
        .axis_iter(Axis(0))
        .zip(test_labels.axis_iter(Axis(0)))
    {
        println!("label: {:?}  {:?}", label, o);
    }

    Ok(())
}
