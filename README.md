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

🔧 `trtyolo-export` is the official ONNX conversion tool for the TensorRT-YOLO project, providing a simple and user-friendly command-line interface to help you convert already-exported YOLO-family ONNX models into TensorRT-YOLO compatible outputs. The converted ONNX files have pre-registered the required TensorRT plugins (including official and custom plugins, supporting detection, segmentation, pose estimation, OBB, etc.), significantly improving model deployment efficiency.

## ✨ Key Features

- **Comprehensive Compatibility**: Supports exported ONNX models from YOLOv3 to YOLO26, as well as model families such as YOLO-World and YOLO-Master, covering object detection, instance segmentation, pose estimation, oriented object detection (OBB), and image classification. See [🖥️ Model Support List](#support-models) for details.
- **Built-in Plugins**: The converted ONNX files have pre-integrated TensorRT official plugins and custom plugins, fully supporting multi-task scenarios such as detection, segmentation, pose estimation, and OBB, greatly simplifying the deployment process.
- **Flexible Configuration**: Provides parameter options such as target opset conversion, threshold tuning, maximum detections, and optional `onnxslim` simplification to meet different deployment requirements.
- **One-Click Conversion**: A concise and intuitive command-line interface with automatic model structure detection, no complex configuration required.

## 🚀 Performance Comparison

<div align="center">

| Model | Official export - Latency 2080Ti TensorRT10 FP16 | trtyolo-export - Latency 2080Ti TensorRT10 FP16 |
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

After installation, you can use the conversion functionality through the `trtyolo-export` command-line tool:

```bash
# View installed version
trtyolo-export --version

# View command help information
trtyolo-export --help

# Convert a basic ONNX model
trtyolo-export -i model.onnx -o output/model-trtyolo.onnx
```

If you need to query the installed version from Python:

```bash
python -c "import trtyolo_export; print(trtyolo_export.__version__)"
```

## 🛠️ Parameter Description

The `trtyolo-export` command supports the following parameters:

<div align="center">

| Parameter | Description | Default Value | Applicable Scenarios |
|-----------|-------------|---------------|----------------------|
| `--version` | Show the installed package version and exit | - | Confirm the CLI version in the current environment |
| `--verbose, --quiet` | Show or hide conversion progress logs | `--verbose` | Control CLI logging verbosity |
| `-i, --input` | Source ONNX file path | - | **Required**, input must be an existing ONNX file |
| `-o, --output` | Converted ONNX output path | - | **Required**, output path must end with `.onnx`; if it matches the input path, `-trtyolo` is appended automatically |
| `--opset` | Target ONNX opset version | Preserve source opset | Convert the converted model to a specific opset before saving |
| `--max-dets` | Maximum detections | 100 | Control the output size when appending TensorRT NMS plugins |
| `--conf-thres` | Confidence threshold | 0.25 | Used by plugin-based and NMS-free postprocess outputs |
| `--iou-thres` | IoU threshold | 0.45 | Used when appending TensorRT NMS plugins |
| `-s, --simplify` | Run `onnxslim` after conversion | False | Slim the converted ONNX model after graph conversion |

</div>

> [!NOTE]
> The input to `trtyolo-export` must already be an exported ONNX model. This tool converts ONNX graphs and postprocess outputs; it does not export directly from `.pt`, `.pth`, `.pdmodel`, or `.pdiparams`.
>
> If `-o/--output` points to the same file as `-i/--input`, the tool will emit a warning and automatically rename the output to `*-trtyolo.onnx` to avoid overwriting the source model.
>
> Official repositories such as [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export), and [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) already provide ONNX export functionality with EfficientNMS plugins. If the official output already matches your deployment requirement, an additional conversion step may be unnecessary.

## 📝 Usage Examples

### Conversion Examples

```bash
# Convert a basic ONNX model
trtyolo-export -i yolov8s.onnx -o output/yolov8s-trtyolo.onnx

# Convert with a target opset
trtyolo-export -i yolov10s.onnx -o output/yolov10s-trtyolo.onnx --opset 12

# Tune TensorRT NMS plugin parameters
trtyolo-export -i yolo11n-obb.onnx -o output/yolo11n-obb-trtyolo.onnx --max-dets 100 --iou-thres 0.45 --conf-thres 0.25

# Simplify the converted ONNX with onnxslim
trtyolo-export -i yolo12n-seg.onnx -o output/yolo12n-seg-trtyolo.onnx -s

# Convert quietly
trtyolo-export --quiet -i model.onnx -o output/model-trtyolo.onnx

# Avoid overwriting the source file
# If -o and -i are the same path, the actual output becomes yolo11n-pose-trtyolo.onnx
trtyolo-export -i yolo11n-pose.onnx -o yolo11n-pose.onnx
```

### TensorRT Engine Construction

The converted ONNX model can be further built into a TensorRT engine using the `trtexec` tool for optimal inference performance:

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

## 📊 Conversion Structure

The converted ONNX model structure is optimized for TensorRT inference and integrates corresponding plugins (official and custom). The model structures for different task types are as follows:

<div>
  <p>
      <img width="100%" src="https://github.com/laugh12321/TensorRT-YOLO/raw/main/assets/model-banner.png"></a>
  </p>
</div>

## 🖥️ Model Support List<div id="support-models"></div>

<div align="center">

| YOLO Series | Source Repo | Detect | Segment | Classify | Pose | OBB |
|-------------|-------------|--------|---------|----------|------|-----|
| YOLOv3 | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ | ✅ | ✅ | - | - |
| YOLOv3 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | - | - | - | - |
| YOLOv5 | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ | ✅ | ✅ | - | - |
| YOLOv5 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | - | - | - | - |
| YOLOv6 | <a href="https://github.com/meituan/YOLOv6">meituan/YOLOv6</a> | 🟢 | - | - | - | - |
| YOLOv6 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | - | - | - | - |
| YOLOv7 | <a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a> | 🟢 | - | - | - | - |
| YOLOv8 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLOv9 | <a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a> | 🟢 | ✅ | - | - | - |
| YOLOv9 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | ✅ | - | - | - |
| YOLOv10 | <a href="https://github.com/THU-MIG/yolov10">THU-MIG/yolov10</a> | ✅ | - | - | - | - |
| YOLOv10 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | - | - | - | - |
| YOLO11 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO12 | <a href="https://github.com/sunsmarterjie/yolov12">sunsmarterjie/yolov12</a> | ✅ | ✅ | ✅ | - | - |
| YOLO12 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO13 | <a href="https://github.com/iMoonLab/yolov13">iMoonLab/yolov13</a> | ✅ | - | - | - | - |
| YOLO26 | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | ✅ | ✅ | ✅ | ✅ |
| YOLO-World | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | - | - | - | - |
| YOLOE | <a href="https://github.com/THU-MIG/yoloe">THU-MIG/yoloe</a> | ✅ | ✅ | - | - | - |
| YOLOE | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ | ✅ | - | - | - |
| YOLO-Master | <a href="https://github.com/isLinXu/YOLO-Master">isLinXu/YOLO-Master</a> | ✅ | ✅ | ✅ | - | - |


</div>

>
> **Symbol Explanation**: `✅` means `trtyolo-export` can convert and can inference | `🟢` means the upstream or repository export path can be used directly for inference | `-` means this task is not provided | `❎` means not supported

## ❓ Frequently Asked Questions

### 1. Why do some models require referring to official export tutorials?

Official repositories for models like YOLOv6, YOLOv7, and YOLOv9 already provide ONNX export functionality with EfficientNMS plugins. If those official ONNX files already satisfy your deployment requirements, you may not need an extra conversion step.

### 2. Which conversion parameters should be adjusted first?

- `--max-dets` limits the number of final detections produced by appended TensorRT NMS plugins
- `--conf-thres` filters low-confidence predictions in plugin-based and NMS-free outputs
- `--iou-thres` controls overlap suppression when TensorRT NMS plugins are appended
- `-s, --simplify` runs `onnxslim`; if your downstream toolchain is sensitive, retry without it
- `--opset` is only needed when your downstream runtime requires a specific ONNX opset version

### 3. What to do if errors are encountered during the conversion process?

- Ensure that the correct version of dependent libraries is installed in your environment
- Check that the input ONNX file exists and the output path ends with `.onnx`
- Confirm that the exported ONNX graph you are using is in the support list
- If opset conversion fails, retry without `--opset` or choose a compatible version
- For models with custom graph modifications, ensure the exported ONNX structure still matches one of the supported conversion patterns
