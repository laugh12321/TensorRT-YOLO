[English](README.en.md) | 简体中文

# 图像分类推理示例

本示例以 yolo11n-cls 模型为例，展示如何使用命令行界面（CLI）、Python 和 C++ 三种方式进行图像分类推理。

[yolo11n-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt)，[【测试图片】ImageNet-part.zip](https://www.ilanzou.com/s/1UsyPhED)

请通过提供的链接下载所需的 `yolo11n-cls.pt` 模型文件和测试图片，并将模型文件保存至 `models` 文件夹，测试图片解压后存放至 `images` 文件夹。

## 模型导出

> [!IMPORTANT]
>
> 如果您仅想通过 `tensorrt_yolo` 提供的命令行界面（CLI）工具 `trtyolo`，导出可供该项目推理的 ONNX 模型（带 TensorRT 插件），可以通过 [PyPI](https://pypi.org/project/tensorrt-yolo) 安装，只需执行以下命令即可：
>
> ```bash
> pip install -U tensorrt_yolo
> ```
> 
> 如果想体验与 C++ 同样的推理速度，则请参考 [安装-tensorrt_yolo](../../docs/cn/build_and_install.md#安装-tensorrt_yolo) 自行构建最新版本的 `tensorrt_yolo`。

使用以下命令导出 ONNX 格式模型，详细的 `trtyolo` CLI 导出方法请阅读 [模型导出](../../docs/cn/model_export.md)：

```bash
trtyolo export -w models/yolo11n-cls.pt -v yolo11 -o models --imgsz 224 -s
```

运行上述命令后，`models` 文件夹中将生成一个 `batch_size` 为 1 的 `yolo11n-cls.onnx` 文件。接下来，使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT 引擎（fp16）：

```bash
trtexec --onnx=models/yolo11n-cls.onnx --saveEngine=models/yolo11n-cls.engine --fp16
```

## 模型推理

> [!IMPORTANT]
>
> 通过 [PyPI](https://pypi.org/project/tensorrt-yolo) 安装的 `tensorrt_yolo` 仅提供可供该项目推理的 ONNX 模型功能，不提供推理功能。
> 如果想体验与 C++ 同样的推理速度，则请参考 [安装-tensorrt_yolo](../../docs/cn/build_and_install.md#安装-tensorrt_yolo) 自行构建最新版本的 `tensorrt_yolo`。

### 使用 CLI 进行推理

1. 使用 `tensorrt_yolo` 库的 `trtyolo` 命令行工具进行推理。运行以下命令查看帮助信息：

    ```bash
    trtyolo infer --help
    ```

2. 运行以下命令进行推理：

    ```bash
    trtyolo infer -e models/yolo11n-cls.engine -m 0 -i images -o output -l labels.txt
    ```

    推理结果将保存至 `output` 文件夹，并生成可视化结果。

### 使用 Python 进行推理

1. 使用 `tensorrt_yolo` 库运行示例脚本 `classify.py`进行推理。
2. 运行以下命令进行推理：

    ```bash
    python classify.py -e models/yolo11n-cls.engine -i images -o output -l labels.txt
    ```

### 使用 C++ 进行推理

1. 确保已按照 [`TensorRT-YOLO` 编译](../../docs/cn/build_and_install.md##rensorrt-yolo-编译) 对项目进行编译。
2. 将 `classify.cpp` 编译为可执行文件：

    ```bash
    # 使用 xmake 编译
    xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy="/path/to/your/TensorRT-YOLO"
    xmake -P . -r

    # 使用 cmake 编译
    cmake -S . -B build -DTENSORRT_PATH="/path/to/your/TensorRT" -DDEPLOY_PATH="/path/to/your/TensorRT-YOLO"
    cmake --build build -j8 --config Release
    ```

    编译完成后，可执行文件将生成在项目根目录的 `bin` 文件夹中。

3. 使用以下命令运行推理：

    ```bash
    cd bin
    ./classify -e ../models/yolo11n-cls.engine -i ../images -o ../output -l ../labels.txt
    ```

通过以上方式，您可以顺利完成模型推理。
