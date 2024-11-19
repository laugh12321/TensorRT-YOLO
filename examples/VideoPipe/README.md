[English](README.en.md) | 简体中文

# 视频分析示例

本示例以 YOLO11n 模型为例，演示如何将 TensorRT-YOLO 的 Deploy 模块集成到 [VideoPipe](https://github.com/sherlockchou86/VideoPipe) 中进行视频分析。

[yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，[demo0.mp4](https://www.ilanzou.com/s/yhUyq8f3)，[demo1.mp4](https://www.ilanzou.com/s/aIhyq8ET)

请通过提供的链接下载所需的 `yolo11n.pt` 模型文件和测试视频，并均保存至 `worksapce` 文件夹。

## 模型导出

使用以下命令导出带 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件的 ONNX 格式，详细的 `trtyolo` CLI 导出方法请阅读 [模型导出](../../docs/cn/model_export.md)：

```bash
trtyolo export -w workspace/yolo11n.pt -v yolo11 -o workspace -b 2 -s
```

运行上述命令后，`models` 文件夹中将生成一个 `batch_size` 为 2 的 `yolo11n.onnx` 文件。接下来，使用 `trtexec` 工具将 ONNX 文件转换为 TensorRT 引擎（fp16）：

```bash
trtexec --onnx=workspace/yolo11n.onnx --saveEngine=workspace/yolo11n.engine --fp16
```

## 项目运行

1. 确保已按照 [`TensorRT-YOLO` 编译](../../docs/cn/build_and_install.md##rensorrt-yolo-编译) 和 [`VideoPipe` 编译和调试](https://github.com/sherlockchou86/VideoPipe/blob/master/README_CN.md#52-编译和调试) 对项目进行编译。

2. 将项目编译为可执行文件：

    ```bash
    # 使用 xmake 编译
    xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy="/path/to/your/TensorRT-YOLO" --videopipe="/path/to/your/VideoPipe"
    xmake -P . -r

    # 使用 cmake 编译
    mkdir -p build && cd build
    cmake -DTENSORRT_PATH="/path/to/your/TensorRT" -DDEPLOY_PATH="/path/to/your/TensorRT-YOLO" -DVIDEOPIPE_PATH="/path/to/your/VideoPipe" .. 
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
