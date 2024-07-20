[English](README.en.md) | 简体中文

# 模型推理示例

在这个示例中，我们以 YOLOv8s 模型为例，演示了如何使用 CLI、Python 和 C++ 三种方式进行模型推理。

## 模型导出

首先，从 [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) 下载 YOLOv8s 模型并保存到 `models` 文件夹中。

然后，使用以下指令将模型导出为带有 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件的 ONNX 格式：

```bash
trtyolo export -w models/yolov8s.pt -v yolov8 -o models
```

执行以上命令后，将在 `models` 文件夹下生成名为 `yolov8s.onnx` 的文件。然后，使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT engine：

```bash
trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s.engine --fp16
```

接下来，我们将使用不同的方式进行模型推理。

## 模型推理

下载 [coco128](https://ultralytics.com/assets/coco128.zip) 数据集，解压缩后，将 `coco128/images/train2017` 文件夹中的图像移动到 `images` 文件夹中以供推理使用。

### 使用 CLI 进行推理

您可以使用 `tensorrt_yolo` 自带的命令行工具 `trtyolo` 进行推理。运行以下命令查看推理相关的帮助信息：

```bash
trtyolo infer --help
```

然后，执行以下命令进行推理：

> 要进一步加速推理过程，请使用 `--cudaGraph` 指令，但此功能仅支持静态模型，不支持动态模型。（4.0之前不支持）

```bash
trtyolo infer -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

此命令将在 `output` 文件夹中生成可视化的推理结果。

### 使用 Python 进行推理

也可以使用 `tensorrt_yolo` 库编写脚本进行推理，`detect.py` 是已经写好的脚本。

> 要进一步加速推理过程，请使用 `--cudaGraph` 指令，但此功能仅支持静态模型，不支持动态模型。

```bash
python detect.py -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

### 使用 C++ 进行推理

使用 C++ 进行推理前，请确保您已按照 [Deploy 编译指南](../../docs/cn/build_and_install.md#deploy-编译) 对 Deploy 进行了编译。

接着，使用 xmake 将 `detect.cpp` 编译为可执行文件：

```bash
xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy=/path/to/your/TensorRT-YOLO

xmake -P . -r
```

在执行上述命令后，将在根目录的 `build` 目录下生成名为 `detect` 的可执行文件。最后，您可以直接运行可执行文件或使用 `xmake run` 命令进行推理。使用 `--help` 查看详细指令选项：

> 要进一步加速推理过程，请使用 `--cudaGraph` 指令，但此功能仅支持静态模型，不支持动态模型。

```bash
xmake run -P . detect -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

以上是进行模型推理的方法示例。
