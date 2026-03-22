[English](README.en.md) | 简体中文

# 视频分析示例

本示例以 YOLO11n 模型为例，演示如何将 TensorRT-YOLO 的 Deploy 模块集成到 [VideoPipe](https://github.com/sherlockchou86/VideoPipe) 中进行视频分析。

[yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，[demo0.mp4](https://www.ilanzou.com/s/yhUyq8f3)，[demo1.mp4](https://www.ilanzou.com/s/aIhyq8ET)

请通过提供的链接下载所需的 `yolo11n.pt` 模型文件和测试视频，并均保存至 `workspace` 文件夹。

## 模型转换

> [!IMPORTANT]
>
> 请先将模型权重导出为 ONNX，再使用项目配套的 [`trtyolo-export`](https://github.com/laugh12321/trtyolo-export) 工具包，将 ONNX 模型转换为兼容 TensorRT-YOLO 推理的输出结构并构建为 TensorRT 引擎。

使用以下命令先导出 ONNX，再转换为适用于该项目推理的模型结构。转换后的 ONNX 会自动集成 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件：

```bash
yolo export model=workspace/yolo11n.pt format=onnx batch=2
trtyolo-export -i workspace/yolo11n.onnx -o workspace/yolo11n-trtyolo.onnx -s
```

运行上述命令后，`workspace` 文件夹中将依次生成原始 ONNX 文件 `yolo11n.onnx` 和转换后的 `yolo11n-trtyolo.onnx`。接下来，使用 `trtexec` 工具将转换后的 ONNX 文件构建为 TensorRT 引擎（fp16）：

```bash
trtexec --onnx=workspace/yolo11n-trtyolo.onnx --saveEngine=workspace/yolo11n.engine --fp16
```

## 项目运行

1. 确保已按照项目文档 和 [`VideoPipe` 编译和调试](https://github.com/sherlockchou86/VideoPipe/blob/master/README_CN.md#52-编译和调试) （只需要执行默认的五个步骤，不需要追加其他编译选项）对项目进行编译。

2. 将项目编译为可执行文件：

    ```bash
    cmake -S . -B build -D VIDEOPIPE_PATH="/path/to/your/VideoPipe"
    cmake --build . -j8 --config Release
    ```

    编译完成后，可执行文件将生成在项目根目录的 `workspace` 文件夹中。

3. 使用以下命令运行推理：

    ```bash
    cd workspace
    ./PipeDemo
    ```

<div align="center">
    <p>
        <img width="100%" src="../../assets/videopipe.png">
    </p>
</div>
