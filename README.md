# RyzenAI with Rust and ONNX Runtime (ORT) Guide

This project demonstrates how to use Rust with the ONNX Runtime (ORT) library to interact with RyzenAI models.

## Project Introduction

### Setting Up the Environment
[准备运行环境]

1. Install NPU Driver and RyzenAI Environment Library
[安装NPU驱动和RyzenAI环境库]

[Installation Instructions](https://ryzenai.docs.amd.com/en/latest/inst.html)

2. Copy and Initialize Runtime Environment
[复制并初始化运行环境]
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

### 1. ResNet Example
[ResNet 示例]
This example demonstrates how to use ONNX Runtime to load a ResNet model and perform inference on the CIFAR-10 dataset.

Reference: [Getting Started Example](https://github.com/amd/RyzenAI-SW/tree/main/tutorial/getting_started_resnet)

```sh
cd resnet
cargo run
```

### 2. YOLOv8 Example
[YOLOv8 示例]

1. Install VcPkg
[安装VcPkg]

* Set Up VCPKG_ROOT Environment Variable
[配置VCPKG_ROOT环境变量]
```powershell
$env:VCPKG_ROOT = "C:\Users\PC\vcpkg"
$env:VCPKGRS_DYNAMIC=1
$env:PATH = "$env:VCPKG_ROOT;$env:PATH"
```

* Install SDL3
[安装SDL3]
```powershell
vcpkg install sdl3
```

2. Download Model
[下载模型]
```sh
cd yolov8/model
python download.py
```

3. Run YOLOv8
[运行YOLOv8]
```sh
cd yolov8
cargo run --release
```
