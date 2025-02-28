# RyzenAI with Rust and ONNX Runtime (ORT)

This project demonstrates how to use Rust with the ONNX Runtime (ORT) library to interact with RyzenAI models.

## 项目介绍

本项目展示了如何使用Rust语言结合ONNX Runtime (ORT)库来调用RyzenAI模型。


### Prepare Runtime Environment
```sh
cd runtime
init.bat
```

### ResNet Demo
This demo shows how to use the ONNX Runtime to load a ResNet model and run inference on the CIFAR-10 dataset.
```sh
cd resnet
cargo run
```