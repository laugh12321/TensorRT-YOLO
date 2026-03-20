[English](README.md) | 简体中文

<div align="center">
  <img width="75%" src="https://github.com/laugh12321/TensorRT-YOLO/raw/main/assets/logo.png">

  <p align="center">
      <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge&color=0074d9"></a>
      <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=yellow">
      <a href="https://pypi.org/project/trtyolo-export/"><img alt="PyPi Version" src="https://img.shields.io/pypi/v/trtyolo-export?style=for-the-badge"></a>
      <img alt="NVIDIA" src="https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white">
  </p>

</div>

---

🔧 `trtyolo-export` 是 TensorRT-YOLO 项目的官方 ONNX 转换工具，提供简单易用的命令行界面，帮助您将已经导出的 YOLO 系列 ONNX 模型转换为兼容 TensorRT-YOLO 推理的输出结构。转换后的 ONNX 文件已预先注册所需 TensorRT 插件（包括官方插件和自定义插件，支持检测、分割、姿态估计、OBB 等任务），大幅提升模型部署效率。

## ✨ 主要特性

- **全面兼容**：支持 YOLOv3 至 YOLO26，以及 YOLO-World、YOLO-Master 等多种模型家族导出的 ONNX 模型，覆盖目标检测、实例分割、姿态估计、旋转框检测、图像分类等任务类型，详见 [🖥️ 模型支持列表](#support-models)
- **插件内置**：转换后的 ONNX 文件已预先集成 TensorRT 官方插件与自定义插件，全面支持检测、分割、姿态估计、OBB 等多任务场景，大幅简化部署流程
- **灵活配置**：提供目标 opset 转换、阈值调节、最大检测数以及可选 `onnxslim` 简化等参数，满足不同部署场景需求
- **一键转换**：命令行界面简洁直观，并支持自动识别模型结构，无需复杂配置

## 🚀 性能对比

<div align="center">

| Model | Official export - Latency 2080Ti TensorRT10 FP16 | trtyolo-export - Latency 2080Ti TensorRT10 FP16 |
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

安装完成后，您可以通过 `trtyolo-export` 命令行工具使用转换功能：

```bash
# 查看当前安装版本
trtyolo-export --version

# 查看命令帮助信息
trtyolo-export --help

# 转换一个基础 ONNX 模型
trtyolo-export -i model.onnx -o output/model-trtyolo.onnx
```

如果您需要在 Python 侧查询当前安装版本：

```bash
python -c "import trtyolo_export; print(trtyolo_export.__version__)"
```

## 🛠️ 参数说明

`trtyolo-export` 命令当前支持以下参数：

<div align="center">

| 参数 | 说明 | 默认值 | 适用场景 |
|------|------|--------|----------|
| `--version` | 显示当前安装包版本并退出 | - | 确认当前环境中的 CLI 版本 |
| `--verbose, --quiet` | 显示或关闭转换过程日志 | `--verbose` | 控制命令行日志输出 |
| `-i, --input` | 输入 ONNX 文件路径 | - | **必需**，输入必须是已存在的 ONNX 文件 |
| `-o, --output` | 输出 ONNX 文件路径 | - | **必需**，输出路径必须以 `.onnx` 结尾；如果与输入路径一致，会自动追加 `-trtyolo` 后缀 |
| `--opset` | 目标 ONNX opset 版本 | 保持输入模型 opset | 在保存前将转换后的模型转换到指定 opset |
| `--max-dets` | 最大检测框数量 | 100 | 控制追加 TensorRT NMS 插件后的输出大小 |
| `--conf-thres` | 置信度阈值 | 0.25 | 用于插件后处理和无 NMS 后处理输出 |
| `--iou-thres` | IoU 阈值 | 0.45 | 用于追加 TensorRT NMS 插件时的重叠抑制 |
| `-s, --simplify` | 转换后执行 `onnxslim` 简化 | False | 在图转换后压缩 ONNX 结构 |

</div>

> [!NOTE]
> `trtyolo-export` 的输入必须是已经导出的 ONNX 模型。该工具负责转换 ONNX 图和后处理输出，不直接从 `.pt`、`.pth`、`.pdmodel` 或 `.pdiparams` 导出模型。
>
> 如果 `-o/--output` 与 `-i/--input` 指向同一个文件，工具会输出一条 warning 日志，并自动将输出文件改名为 `*-trtyolo.onnx`，以避免覆写源模型。
>
> 官方仓库如 [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800)、[YOLOv7](https://github.com/WongKinYiu/yolov7#export)、[YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) 已经提供了带有 EfficientNMS 插件的 ONNX 导出功能。如果官方导出结果已经满足您的部署需求，通常不需要额外转换。

## 📝 使用示例

### 转换示例

```bash
# 转换一个基础 ONNX 模型
trtyolo-export -i yolov8s.onnx -o output/yolov8s-trtyolo.onnx

# 指定目标 opset 版本
trtyolo-export -i yolov10s.onnx -o output/yolov10s-trtyolo.onnx --opset 12

# 调整 TensorRT NMS 插件参数
trtyolo-export -i yolo11n-obb.onnx -o output/yolo11n-obb-trtyolo.onnx --max-dets 100 --iou-thres 0.45 --conf-thres 0.25

# 使用 onnxslim 简化转换后的 ONNX
trtyolo-export -i yolo12n-seg.onnx -o output/yolo12n-seg-trtyolo.onnx -s

# 安静模式下转换
trtyolo-export --quiet -i model.onnx -o output/model-trtyolo.onnx

# 避免覆写源文件
# 如果 -o 和 -i 相同，实际输出会变成 yolo11n-pose-trtyolo.onnx
trtyolo-export -i yolo11n-pose.onnx -o yolo11n-pose.onnx
```

### TensorRT 引擎构建

转换后的 ONNX 模型可以通过 `trtexec` 工具进一步构建为 TensorRT 引擎，以获得最佳推理性能：

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

## 📊 转换结构

转换后的 ONNX 模型结构针对 TensorRT 推理进行了优化，并集成了相应插件（官方插件和自定义插件）。不同任务类型的模型结构如下：

<div>
  <p>
      <img width="100%" src="https://github.com/laugh12321/TensorRT-YOLO/raw/main/assets/model-banner.png"></a>
  </p>
</div>

## 🖥️ 模型支持列表<div id="support-models"></div>

<div align="center">

| YOLO 系列 | 来源仓库 | Detect | Segment | Classify | Pose | OBB |
|-----------|----------|--------|---------|----------|------|-----|
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
> **符号说明**：`✅` 表示 `trtyolo-export` 可转换且可推理 | `🟢` 表示上游或仓库导出路径可直接用于推理 | `-` 表示该任务不提供 | `❎` 表示不支持

## ❓ 常见问题

### 1. 为什么有些模型需要参考官方导出教程？

YOLOv6、YOLOv7、YOLOv9 等模型的官方仓库已经提供了带有 EfficientNMS 插件的 ONNX 导出功能。如果官方导出结果已经满足部署需求，通常不需要额外执行转换步骤。

### 2. 哪些转换参数最值得优先调整？

- `--max-dets` 用于限制追加 TensorRT NMS 插件后的最终检测框数量
- `--conf-thres` 用于过滤低置信度预测
- `--iou-thres` 用于控制追加 TensorRT NMS 插件时的重叠抑制强度
- `-s, --simplify` 会执行 `onnxslim`，如果下游工具链对模型较敏感，可以先关闭它再重试
- `--opset` 仅在下游运行时要求特定 ONNX opset 版本时才需要显式指定

### 3. 转换过程中遇到错误怎么办？

- 确保您的环境中已安装正确版本的依赖库
- 检查输入 ONNX 文件路径是否存在，以及输出路径是否以 `.onnx` 结尾
- 确认您使用的导出 ONNX 图结构在支持列表中
- 如果 opset 转换失败，请去掉 `--opset` 后重试，或改用兼容的 opset 版本
- 对于做过自定义图修改的模型，请确认导出的 ONNX 结构仍能匹配当前工具支持的转换模式
