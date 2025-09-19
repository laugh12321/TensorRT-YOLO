[简体中文](README.cn.md) | English

<div align="center">
  <img width="75%" src="https://github.com/laugh12321/TensorRT-YOLO/raw/main/assets/logo.png">

  <p align="center">
      <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge&color=0074d9"></a>
      <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge&color=0074d9"></a>
      <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=3dd3ff">
      <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black">
      <img alt="Arch" src="https://img.shields.io/badge/Arch-x86%20%7C%20ARM-0091BD?style=for-the-badge&logo=cpu&logoColor=white">
      <img alt="NVIDIA" src="https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white">
  </p>

</div>

---

🔧 `trtyolo-export` is the official export tool for the TensorRT-YOLO project, providing a simple and user-friendly command-line interface to help you export various YOLO series models to TensorRT-YOLO compatible ONNX format with just one click. The exported ONNX files have pre-registered the required TensorRT plugins (including official and custom plugins, supporting detection, segmentation, pose estimation, OBB, etc.), significantly improving model deployment efficiency.

## ✨ Key Features

- **Comprehensive Compatibility**: Supports the full range of models from YOLOv3 to YOLO12, as well as various variants like PP-YOLOE, PP-YOLOE+, YOLO-World, and YOLOE, covering all five core visual task types: object detection, instance segmentation, pose estimation, oriented object detection (OBB), and image classification, fully meeting diverse application needs. See [🖥️ Model Support List](#support-models) for details.
- **Built-in Plugins**: The exported ONNX files have pre-integrated TensorRT official plugins and custom plugins, fully supporting multi-task scenarios such as detection, segmentation, pose estimation, and OBB, greatly simplifying the deployment process.
- **Flexible Configuration**: Provides rich parameter options such as dynamic batch size, custom thresholds, and image dimensions to meet different scenario requirements.
- **One-Click Operation**: A concise and intuitive command-line interface, no complex configuration required, enabling quick model export.

## 🚀 Performance Comparison

<div align="center">

| Model | Official export - Latency 2080Ti TensorRT10 FP16 | trtyolo export - Latency 2080Ti TensorRT10 FP16 |
|:-----:|:-----------------------:|:----------------------:|
| YOLOv11N | 1.611 ± 0.061        | 1.428 ± 0.097          |
| YOLOv11S | 2.055 ± 0.147        | 1.886 ± 0.145          |
| YOLOv11M | 3.028 ± 0.167        | 2.865 ± 0.235          |
| YOLOv11L | 3.856 ± 0.287        | 3.682 ± 0.309          |
| YOLOv11X | 6.377 ± 0.487        | 6.195 ± 0.482          |

</div>

## 💨 Quick Start

### Installation

#### 📦 Recommended Method: Install via pip

In a Python>=3.8 environment, you can quickly install the `trtyolo-export` package via pip:

```bash
pip install trtyolo-export
```

#### 🔧 Alternative Method: Build from Source

If you need the latest development version or want to make custom modifications, you can build from source:

```bash
# Clone the repository (if you don't have a local copy yet)
git clone https://github.com/laugh12321/TensorRT-YOLO

# Enter the project directory (assuming you're already in this directory)
cd TensorRT-YOLO

# Switch to the export branch
git checkout export

# Build and install
pip install build
python -m build
pip install dist/*.whl
```

### Basic Usage

After installation, you can use the export functionality through the `trtyolo` command-line tool:

```bash
# View export command help information
trtyolo export --help

# Export a basic YOLO model
trtyolo export -v yolov8 -w yolov8s.pt -o output
```

## 🛠️ Parameter Description

The `trtyolo export` command supports various parameters to meet different scenario requirements:

<div align="center">

| Parameter | Description | Default Value | Applicable Scenarios |
|-----------|-------------|---------------|----------------------|
| `-v, --version` | Model version | - | **Required**, specify the type of model to export |
| `-o, --output` | Export model save directory | - | **Required**, specify the output folder path |
| `-w, --weights` | PyTorch weight file path | - | **Required** for non-PP-YOLOE models |
| `--model_dir` | PP-YOLOE model directory | - | **Required** for PP-YOLOE models |
| `--model_filename` | PP-YOLOE model filename | - | **Required** for PP-YOLOE models |
| `--params_filename` | PP-YOLOE parameter filename | - | **Required** for PP-YOLOE models |
| `-b, --batch` | Batch size (-1 for dynamic) | 1 | Adjust the batch processing capability of the model |
| `--max_boxes` | Maximum number of detection boxes | 100 | (Not applicable to classification models) Control the maximum number of detection boxes output per image |
| `--iou_thres` | NMS IoU threshold | 0.45 | Control the overlap threshold for filtering detection boxes |
| `--conf_thres` | Confidence threshold | 0.25 | Filter low-confidence detection results |
| `--imgsz` | Image size (height,width) | 640 | Set the input image size of the model |
| `--names` | Custom class names (comma-separated) | - | Only applicable to YOLO-World and YOLOE models |
| `--repo_dir` | Local repository directory | - | Only applicable to YOLOv3 and YOLOv5 models |
| `--opset` | ONNX opset version | 12 | Specify the ONNX operator set version |
| `-s, --simplify` | Whether to simplify the ONNX model | False | Simplify the model structure and reduce model size |

</div>

> [!NOTE]
> When exporting ONNX models for PP-YOLOE and PP-YOLOE+, only the batch dimension will be adjusted, while the height and width dimensions remain unchanged. You need to make relevant settings in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), with the default value usually being 640.
>
> Official repositories such as [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export), and [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) already provide ONNX model export functionality with EfficientNMS plugins, so they are not repeated here.

## 📝 Usage Examples

### Export Examples

```bash
# Export a YOLOv3 model from a remote repository
trtyolo export -v yolov3 -w yolov3.pt -o output

# Export a YOLOv5 Classify model from a local repository
trtyolo export -v yolov5 -w yolov5s-cls.pt -o output --repo_dir your_local_yolovs_repository

# Export a YOLO series model trained with Ultralytics (YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv9, YOLOv10, YOLOv11, etc.) with plugin parameters and dynamic batch
trtyolo export -v ultralytics -w yolov8s.pt -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# Export PP-YOLOE and PP-YOLOE+ models
trtyolo export -v pp-yoloe --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output

# Export YOLOv10 model with height 1080 and width 1920
trtyolo export -v yolov10 -w yolov10s.pt -o output --imgsz 1080,1920

# Export YOLO11 OBB model
trtyolo export -v yolo11 -w yolo11n-obb.pt -o output

# Export YOLO12 Segment model
trtyolo export -v yolo12 -w yolo12n-seg.pt -o output

# Export YOLO-World model with custom classes
trtyolo export -v yolo-world -w yoloworld.pt -o output --names "person,car,dog"

# Export YOLOE model
trtyolo export -v yoloe -w yoloe.pt -o output
```

### TensorRT Engine Construction

The exported ONNX model can be further built into a TensorRT engine using the `trtexec` tool for optimal inference performance:

```bash
# Static batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# Dynamic batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

# ! Note: For segmentation, pose estimation, and OBB models, you need to specify staticPlugins and setPluginsToSerialize parameters to ensure correct loading of custom plugins compiled by the project

# YOLOv8-OBB static batch
trtexec --onnx=yolov8n-obb.onnx --saveEngine=yolov8n-obb.engine --fp16 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so

# YOLO11-OBB dynamic batch
trtexec --onnx=yolo11n-obb.onnx --saveEngine=yolo11n-obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/custom_plugins.dll --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/custom_plugins.dll
```

## 📊 Export Structure

The exported ONNX model structure is optimized for TensorRT inference and integrates corresponding plugins (official and custom). The model structures for different task types are as follows:

<div>
  <p>
      <img width="100%" src="https://github.com/laugh12321/TensorRT-YOLO/raw/main/assets/model-banner.png"></a>
  </p>
</div>

## 🖥️ Model Support List<div id="support-models"></div>

<div align="center">

| Task Scenario | Model | CLI Export | Inference Deployment |
|---------------|-------|------------|----------------------|
| **Detect** | | | |
| | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/meituan/YOLOv6">meituan/YOLOv6</a> | ❎ Refer to <a href="https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800">official export tutorial</a> | ✅ Supported |
| | <a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a> | ❎ Refer to <a href="https://github.com/WongKinYiu/yolov7#export">official export tutorial</a> | ✅ Supported |
| | <a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a> | ❎ Refer to <a href="https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461">official export tutorial</a> | ✅ Supported |
| | <a href="https://github.com/THU-MIG/yolov10">THU-MIG/yolov10</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/sunsmarterjie/yolov12">sunsmarterjie/yolov12</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/ultralytics">YOLO-World V2</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/THU-MIG/yoloe">THU-MIG/yoloe</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/PaddlePaddle/PaddleDetection">PaddleDetection/PP-YOLOE+</a> | ✅ Supported | ✅ Supported |
| **Segment** | | | |
| | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/meituan/YOLOv6/tree/yolov6-seg">meituan/YOLOv6-seg</a> | ❎ Need to refer to code for self-implementation | 🟢 Inference possible |
| | <a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a> | ❎ Need to refer to code for self-implementation | 🟢 Inference possible |
| | <a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a> | ❎ Need to refer to code for self-implementation | 🟢 Inference possible |
| | <a href="https://github.com/THU-MIG/yoloe">THU-MIG/yoloe</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ Supported | ✅ Supported |
| **Classify** | | | |
| | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ Supported | ✅ Supported |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ Supported | ✅ Supported |
| **Pose** | | | |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ Supported | ✅ Supported |
| **OBB** | | | |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ Supported | ✅ Supported |


</div>

> **Symbol Explanation**: ✅ Supported | ❔ In progress | ❎ Not supported yet | 🟢 Inference possible after self-implementation of export

## ❓ Frequently Asked Questions

### 1. Why do some models require referring to official export tutorials?

Official repositories for models like YOLOv6, YOLOv7, and YOLOv9 already provide ONNX model export functionality with EfficientNMS plugins. To avoid redundant development, we recommend using the export methods provided directly by the official repositories.

### 2. How to choose the appropriate batch size?

- For fixed scenarios and hardware, static batch (e.g., `--batch 4`) can be chosen for optimal performance
- For varying input scales, dynamic batch (e.g., `--batch -1`) is recommended, combined with TensorRT dynamic shape functionality
- In actual use, adjustments should be made according to your GPU memory size and inference latency requirements

### 3. What to do if errors are encountered during the export process?

- Ensure that the correct version of dependent libraries is installed in your environment
- Check if the model file path is correct
- Confirm that the model version you are using is in the support list
- For PP-YOLOE models, ensure all necessary files and parameters are provided
- For models with custom modifications based on Ultralytics, ensure that `trtyolo-export` and the custom model are installed in the same Python environment; if this condition cannot be met, ensure that the custom modified code is completely synchronized with the Ultralytics code version that `trtyolo-export` depends on