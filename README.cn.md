[English](README.md) | 简体中文

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

🔧 `trtyolo-export` 是 TensorRT-YOLO 项目的官方导出工具，提供简单易用的命令行界面，帮助您将各种 YOLO 系列模型一键导出为兼容 TensorRT-YOLO 推理的 ONNX 格式。导出的 ONNX 文件已预先注册所需 TensorRT 插件（包括官方插件和自定义插件，支持检测、分割、姿态估计、OBB等任务），大幅提升模型部署效率。

## ✨ 主要特性

- **全面兼容**：支持 YOLOv3 至 YOLO12 全系列模型，以及 PP-YOLOE、PP-YOLOE+、YOLO-World、YOLOE 等多种变体，全面覆盖目标检测、实例分割、姿态估计、旋转框检测、图像分类等视觉五大核心任务类型，充分满足多样化应用需求，详见 [🖥️ 模型支持列表](#support-models)
- **插件内置**：导出的 ONNX 文件已预先集成 TensorRT 官方插件与自定义插件，全面支持检测、分割、姿态估计、OBB 等多任务场景，大幅简化部署流程
- **灵活配置**：提供动态批量大小、自定义阈值、图像尺寸等丰富参数选项，满足不同场景需求
- **一键操作**：简洁直观的命令行界面，无需复杂配置，实现模型快速导出

## 🚀 性能对比

<div align="center">

| Model | Official export - Latency 2080Ti TensorRT10 FP16 | trtyolo export - Latency 2080Ti TensorRT10 FP16 |
|:-----:|:-----------------------:|:----------------------:|
| YOLOv11N | 1.611 ± 0.061        | 1.428 ± 0.097          |
| YOLOv11S | 2.055 ± 0.147        | 1.886 ± 0.145          |
| YOLOv11M | 3.028 ± 0.167        | 2.865 ± 0.235          |
| YOLOv11L | 3.856 ± 0.287        | 3.682 ± 0.309          |
| YOLOv11X | 6.377 ± 0.487        | 6.195 ± 0.482          |

</div>

## 💨 快速开始

### 安装

#### 📦 推荐方式：通过 pip 安装

在 Python>=3.8 环境中，您可以通过 pip 快速安装 `trtyolo-export` 包：

```bash
pip install trtyolo-export
```

#### 🔧 可选方式：从源码构建

如果您需要最新的开发版本或进行自定义修改，可以从源码构建：

```bash
# 克隆仓库（如果您还没有本地副本）
git clone https://github.com/laugh12321/TensorRT-YOLO

# 进入项目目录（假设您已经在这个目录下）
cd TensorRT-YOLO

# 切换至 export 分支
git checkout export

# 构建并安装
pip install build
python -m build
pip install dist/*.whl
```

### 基本用法

安装完成后，您可以通过 `trtyolo` 命令行工具使用导出功能：

```bash
# 查看导出命令帮助信息
trtyolo export --help

# 导出一个基本的 YOLO 模型
trtyolo export -v yolov8 -w yolov8s.pt -o output
```

## 🛠️ 参数说明

`trtyolo export` 命令支持多种参数，以满足不同场景的需求：

<div align="center">

| 参数 | 说明 | 默认值 | 适用场景 |
|------|------|--------|----------|
| `-v, --version` | 模型版本 | - | **必需**，指定要导出的模型类型 |
| `-o, --output` | 导出模型保存目录 | - | **必需**，指定输出文件夹路径 |
| `-w, --weights` | PyTorch 权重文件路径 | - | 非 PP-YOLOE 模型**必需** |
| `--model_dir` | PP-YOLOE 模型目录 | - | PP-YOLOE 模型**必需** |
| `--model_filename` | PP-YOLOE 模型文件名 | - | PP-YOLOE 模型**必需** |
| `--params_filename` | PP-YOLOE 参数文件名 | - | PP-YOLOE 模型**必需** |
| `-b, --batch` | 批量大小 (-1 表示动态) | 1 | 调整模型的批量处理能力 |
| `--max_boxes` | 最大检测框数量 | 100 | （不适用于分类模型）控制每张图像最多输出的检测框数量 |
| `--iou_thres` | NMS IoU 阈值 | 0.45 | 控制检测框过滤的重叠度阈值 |
| `--conf_thres` | 置信度阈值 | 0.25 | 过滤低置信度检测结果 |
| `--imgsz` | 图像尺寸 (高度,宽度) | 640 | 设置模型输入图像的尺寸 |
| `--names` | 自定义类别名称 (逗号分隔) | - | 仅适用于 YOLO-World 和 YOLOE 模型 |
| `--repo_dir` | 本地仓库目录 | - | 仅适用于 YOLOv3 和 YOLOv5 模型 |
| `--opset` | ONNX opset 版本 | 12 | 指定 ONNX 算子集版本 |
| `-s, --simplify` | 是否简化 ONNX 模型 | False | 简化模型结构，减小模型体积 |

</div>

> [!NOTE]
> 在导出 PP-YOLOE 和 PP-YOLOE+ 的 ONNX 模型时，仅会调整 batch 维度，而 height 和 width 维度保持不变。您需要在 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 中进行相关设置，默认值通常为 640。
>
> 官方仓库如 [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800)、[YOLOv7](https://github.com/WongKinYiu/yolov7#export)、[YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) 已经提供了带有 EfficientNMS 插件的 ONNX 模型导出功能，因此此处不再重复提供。

## 📝 使用示例

### 导出示例

```bash
# 导出远程仓库中的 YOLOv3 模型
trtyolo export -v yolov3 -w yolov3.pt -o output

# 导出本地仓库中的 YOLOv5 Classify 模型
trtyolo export -v yolov5 -w yolov5s-cls.pt -o output --repo_dir your_local_yolovs_repository

# 使用 Ultralytics 训练的 YOLO 系列模型（YOLOv3、YOLOv5、YOLOv6、YOLOv8、YOLOv9、YOLOv10、YOLOv11 等），并指定插件参数，以动态 batch 导出
trtyolo export -v ultralytics -w yolov8s.pt -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# 导出 PP-YOLOE 和 PP-YOLOE+ 模型
trtyolo export -v pp-yoloe --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output

# 导出 YOLOv10 模型，高度 1080，宽度 1920
trtyolo export -v yolov10 -w yolov10s.pt -o output --imgsz 1080,1920

# 导出 YOLO11 OBB 模型
trtyolo export -v yolo11 -w yolo11n-obb.pt -o output

# 导出 YOLO12 Segment 模型
trtyolo export -v yolo12 -w yolo12n-seg.pt -o output

# 导出 YOLO-World 模型，并自定义类别
trtyolo export -v yolo-world -w yoloworld.pt -o output --names "person,car,dog"

# 导出 YOLOE 模型
trtyolo export -v yoloe -w yoloe.pt -o output
```

### TensorRT 引擎构建

导出的 ONNX 模型可以通过 `trtexec` 工具进一步构建为 TensorRT 引擎，以获得最佳推理性能：

```bash
# 静态 batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# 动态 batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

# ! 注意：分割、姿态估计、OBB 模型需要指定 staticPlugins 与 setPluginsToSerialize 参数，以确保项目编译的自定义插件正确加载

# YOLOv8-OBB 静态 batch
trtexec --onnx=yolov8n-obb.onnx --saveEngine=yolov8n-obb.engine --fp16 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so

# YOLO11-OBB 动态 batch
trtexec --onnx=yolo11n-obb.onnx --saveEngine=yolo11n-obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/custom_plugins.dll --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/custom_plugins.dll
```

## 📊 导出结构

导出的 ONNX 模型结构针对 TensorRT 推理进行了优化，集成了相应的插件（官方插件和自定义插件）。不同任务类型的模型结构如下：

<div>
  <p>
      <img width="100%" src="https://github.com/laugh12321/TensorRT-YOLO/raw/main/assets/model-banner.png"></a>
  </p>
</div>

## 🖥️ 模型支持列表<div id="support-models"></div>

<div align="center">

| 任务场景 | 模型 | CLI 导出 | 推理部署 |
|----------|------|----------|----------|
| **Detect** | | | |
| | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/meituan/YOLOv6">meituan/YOLOv6</a> | ❎ 参考<a href="https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800">官方导出教程</a> | ✅ 支持 |
| | <a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a> | ❎ 参考<a href="https://github.com/WongKinYiu/yolov7#export">官方导出教程</a> | ✅ 支持 |
| | <a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a> | ❎ 参考<a href="https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461">官方导出教程</a> | ✅ 支持 |
| | <a href="https://github.com/THU-MIG/yolov10">THU-MIG/yolov10</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/sunsmarterjie/yolov12">sunsmarterjie/yolov12</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/ultralytics">YOLO-World V2</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/THU-MIG/yoloe">THU-MIG/yoloe</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/PaddlePaddle/PaddleDetection">PaddleDetection/PP-YOLOE+</a> | ✅ 支持 | ✅ 支持 |
| **Segment** | | | |
| | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/meituan/YOLOv6/tree/yolov6-seg">meituan/YOLOv6-seg</a> | ❎ 需参考代码自行实现 | 🟢 可推理 |
| | <a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a> | ❎ 需参考代码自行实现 | 🟢 可推理 |
| | <a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a> | ❎ 需参考代码自行实现 | 🟢 可推理 |
| | <a href="https://github.com/THU-MIG/yoloe">THU-MIG/yoloe</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ 支持 | ✅ 支持 |
| **Classify** | | | |
| | <a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a> | ✅ 支持 | ✅ 支持 |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ 支持 | ✅ 支持 |
| **Pose** | | | |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ 支持 | ✅ 支持 |
| **OBB** | | | |
| | <a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a> | ✅ 支持 | ✅ 支持 |


</div>

> **符号说明**：✅ 已支持 | ❔ 进行中 | ❎ 暂不支持 | 🟢 可自行实现导出后推理

## ❓ 常见问题

### 1. 为什么有些模型需要参考官方导出教程？

YOLOv6、YOLOv7、YOLOv9 等模型的官方仓库已经提供了带有 EfficientNMS 插件的 ONNX 模型导出功能，为避免重复开发，我们建议直接使用官方提供的导出方法。

### 2. 如何选择合适的 batch 大小？

- 对于固定场景和硬件，可选择静态 batch (如 `--batch 4`) 获得最佳性能
- 对于变化的输入规模，建议使用动态 batch (如 `--batch -1`)，配合 TensorRT 动态 shape 功能
- 实际使用中，应根据您的 GPU 显存大小和推理延迟要求进行调整

### 3. 导出过程中遇到错误怎么办？

- 确保您的环境中已安装正确版本的依赖库
- 检查模型文件路径是否正确
- 确认您使用的模型版本在支持列表中
- 对于 PP-YOLOE 模型，确保提供了所有必需的文件和参数
- 对于基于 Ultralytics 进行自定义修改的模型，请确保 `trtyolo-export` 与该自定义模型安装在同一 Python 环境中；若无法满足此条件，则需确保自定义修改的代码与 `trtyolo-export` 依赖的 Ultralytics 代码版本保持完全同步
