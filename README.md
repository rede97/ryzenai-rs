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

* Linux env
```sh
cd runtime
./init.sh
source env.sh
```

### 1. ResNet Demo
This demo shows how to use the ONNX Runtime to load a ResNet model and run inference on the CIFAR-10 dataset.

Reference: [Getting Started Example](https://github.com/amd/RyzenAI-SW/tree/main/tutorial/getting_started_resnet)

```sh
cd resnet
cargo run
```

### 2. Yolov8 Demo

* Download Model
```sh
cd yolov8/model
python download.py
```

* Runing Yolov8
```sh
cd yolov8
cargo run
```


