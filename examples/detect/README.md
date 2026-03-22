[English](README.en.md) | 简体中文

# 目标检测推理示例

本示例以 yolo11n 模型为例，展示如何使用命令行界面（CLI）、Python 和 C++ 三种方式进行目标检测推理。

[yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，[【测试图片】COCO-part.zip](https://www.ilanzou.com/s/N5Oyq8hZ)

请通过提供的链接下载所需的 `yolo11n.pt` 模型文件和测试图片，并将模型文件保存至 `models` 文件夹，测试图片解压后存放至 `images` 文件夹。

## 模型转换

> [!IMPORTANT]
>
> 请先将模型权重导出为 ONNX，再使用项目配套的 [`trtyolo-export`](https://github.com/laugh12321/trtyolo-export) 工具包，将 ONNX 模型转换为兼容 TensorRT-YOLO 推理的输出结构并构建为 TensorRT 引擎。

使用以下命令先导出 ONNX，再转换为适用于该项目推理的模型结构。转换后的 ONNX 会自动集成 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件：

```bash
yolo export model=models/yolo11n.pt format=onnx batch=1
trtyolo-export -i models/yolo11n.onnx -o models/yolo11n-trtyolo.onnx -s
```

运行上述命令后，`models` 文件夹中将依次生成原始 ONNX 文件 `yolo11n.onnx` 和转换后的 `yolo11n-trtyolo.onnx`。接下来，使用 `trtexec` 工具将转换后的 ONNX 文件构建为 TensorRT 引擎（fp16）：

```bash
trtexec --onnx=models/yolo11n-trtyolo.onnx --saveEngine=models/yolo11n.engine --fp16
```

## 模型推理

### 使用 Python 进行推理

1. 使用 `trtyolo` 库运行示例脚本 `detect.py`进行推理。
2. 运行以下命令进行推理：

    ```bash
    python detect.py -e models/yolo11n.engine -i images -o output -l labels.txt
    ```

### 使用 C++ 进行推理

1. 确保已按照项目文档对项目进行编译。
2. 将 `detect.cpp` 编译为可执行文件：

    ```bash
    cmake -S . -B build
    cmake --build build -j8 --config Release
    ```

    编译完成后，可执行文件将生成在项目根目录的 `bin` 文件夹中。

3. 使用以下命令运行推理：

    ```bash
    cd bin
    ./detect -e ../models/yolo11n.engine -i ../images -o ../output -l ../labels.txt
    ```

通过以上方式，您可以顺利完成模型推理。
