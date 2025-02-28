# RyzenAI with Rust and ONNX Runtime (ORT)

This project demonstrates how to use Rust with the ONNX Runtime (ORT) library to interact with RyzenAI models.

## 项目介绍

本项目展示了如何使用Rust语言结合ONNX Runtime (ORT)库来调用RyzenAI模型。


### Prepare Runtime Environment

1. Intall NPU Driver and ryzen-ai Environment Library

[Installation Instructions](https://ryzenai.docs.amd.com/en/latest/inst.html)

2. Copy runtime
```sh
cd runtime
init.bat
```

### 1. ResNet Demo
This demo shows how to use the ONNX Runtime to load a ResNet model and run inference on the CIFAR-10 dataset.
```sh
cd resnet
cargo run
```