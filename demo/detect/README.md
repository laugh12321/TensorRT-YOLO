[English](README.en.md) | 简体中文

# 模型推理示例

本示例以 YOLOv8s 模型为例，展示如何使用 CLI、Python 和 C++ 三种方式进行模型推理。

> [!IMPORTANT]  
> 如果要使用 [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin)  插件推理 OBB 模型，请参考 [构建 TensorRT 自定义插件](../../docs/cn/build_trt_custom_plugin.md) 进行构建。

## 模型导出

### 检测模型 (Detection)

1. 下载 [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) 模型，并将其保存至 `models` 文件夹。
2. 使用以下命令将模型导出为带有 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件的 ONNX 格式：

    ```bash
    trtyolo export -w models/yolov8s.pt -v yolov8 -o models
    ```

    运行上述命令后，`models` 文件夹中将生成 `yolov8s.onnx` 文件。接下来，使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT 引擎：

    ```bash
    trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s.engine --fp16
    ```

### 旋转边界框模型 (OBB)

1. 下载 [YOLOv8s-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-obb.pt) 模型，并将其保存至 `models` 文件夹。
2. 使用以下命令将模型导出为带有 [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin/) 插件的 ONNX 格式：

    ```bash
    trtyolo export -w models/yolov8s-obb.pt -v yolov8 -o models
    ```

    运行上述命令后，`models` 文件夹中将生成 `yolov8s-obb.onnx` 文件。接下来，使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT 引擎：

    ```bash
    trtexec --onnx=models/yolov8s-obb.onnx --saveEngine=models/yolov8s-obb.engine --fp16
    ```

## 数据集准备

### 检测模型 (Detection)

1. 下载 [coco128](https://ultralytics.com/assets/coco128.zip) 数据集。
2. 解压后，将 `coco128/images/train2017` 文件夹中的图像移动到 `images` 文件夹，以供推理使用。

### 旋转边界框模型 (OBB)

1. 下载 [DOTA-v1.0](https://drive.google.com/file/d/1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK/view) 数据集。
2. 解压后，将 `part1/images` 文件夹中的图像移动到 `images` 文件夹，以供推理使用。

## 模型推理

### 使用 CLI 进行推理

1. 使用 `trtyolo` 命令行工具进行推理。运行以下命令查看帮助信息：

    ```bash
    trtyolo infer --help
    ```

2. 运行以下命令进行推理：

    > [!NOTE] 
    > 从 4.0 版本开始新增的 `--cudaGraph` 指令可以进一步加速推理过程，但该功能仅支持静态模型。
    > 
    > 从 4.2 版本开始，支持 OBB 模型推理，并新增 `-m, --mode` 指令，用于选择 Detection 还是 OBB 模型。

    ```bash
    # 检测模型
    trtyolo infer -e models/yolov8s.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # 旋转边界框模型
    trtyolo infer -e models/yolov8s-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

    推理结果将保存至 `output` 文件夹，并生成可视化结果。

### 使用 Python 进行推理

1. 使用 `tensorrt_yolo` 库进行 Python 推理。示例脚本 `detect.py` 已准备好。
2. 运行以下命令进行推理：

    > [!NOTE] 
    > `--cudaGraph` 指令可以进一步加速推理过程，但该功能仅支持静态模型。

    ```bash
    # 检测模型
    python detect.py -e models/yolov8s.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # 旋转边界框模型
    python detect.py -e models/yolov8s-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

### 使用 C++ 进行推理

1. 确保已按照 [Deploy 编译指南](../../docs/cn/build_and_install.md#deploy-编译) 对项目进行编译。
2. 使用 xmake 将 `detect.cpp` 编译为可执行文件：

    ```bash
    xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy="/path/to/your/TensorRT-YOLO"

    xmake -P . -r
    ```

    编译完成后，可执行文件将生成在项目根目录的 `build` 文件夹中。

3. 使用以下命令运行推理：

    > [!NOTE] 
    > `--cudaGraph` 指令可以进一步加速推理过程，但该功能仅支持静态模型。

    ```bash
    # 检测模型
    xmake run -P . detect -e models/yolov8s.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # 旋转边界框模型
    xmake run -P . detect -e models/yolov8s-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

通过以上方式，您可以顺利完成模型推理。

