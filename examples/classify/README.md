[English](README.en.md) | 简体中文

# 图像分类推理示例

本示例以 yolo11n-cls 模型为例，展示如何使用命令行界面（CLI）、Python 和 C++ 三种方式进行图像分类推理。

[yolo11n-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt)，[【测试图片】ImageNet-part.zip](https://www.ilanzou.com/s/1UsyPhED)

请通过提供的链接下载所需的 `yolo11n-cls.pt` 模型文件和测试图片，并将模型文件保存至 `models` 文件夹，测试图片解压后存放至 `images` 文件夹。

## 模型导出

> [!IMPORTANT]
>
> 使用项目配套的 [`trtyolo-export`](https://github.com/laugh12321/TensorRT-YOLO/tree/export) 工具包，导出适用于该项目推理的 ONNX 模型并构建为 TensorRT 引擎。

使用以下命令导出 ONNX 格式模型：

```bash
trtyolo export -w models/yolo11n-cls.pt -v yolo11 -o models --imgsz 224 -s
```

运行上述命令后，`models` 文件夹中将生成一个 `batch_size` 为 1 的 `yolo11n-cls.onnx` 文件。接下来，使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT 引擎（fp16）：

```bash
trtexec --onnx=models/yolo11n-cls.onnx --saveEngine=models/yolo11n-cls.engine --fp16
```

## 模型推理

### 使用 Python 进行推理

1. 使用 `trtyolo` 库运行示例脚本 `classify.py`进行推理。
2. 运行以下命令进行推理：

    ```bash
    python classify.py -e models/yolo11n-cls.engine -i images -o output -l labels.txt
    ```

### 使用 C++ 进行推理

1. 确保已按照文档对项目进行编译。
2. 将 `classify.cpp` 编译为可执行文件：

    ```bash
    cmake -S . -B build
    cmake --build build -j8 --config Release
    ```

    编译完成后，可执行文件将生成在项目根目录的 `bin` 文件夹中。

3. 使用以下命令运行推理：

    ```bash
    cd bin
    ./classify -e ../models/yolo11n-cls.engine -i ../images -o ../output -l ../labels.txt
    ```

通过以上方式，您可以顺利完成模型推理。
