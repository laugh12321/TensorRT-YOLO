English | [简体中文](README.md)

# Video Analysis Example

This example uses the YOLO11n model to demonstrate how to integrate the TensorRT-YOLO Deploy module into [VideoPipe](https://github.com/sherlockchou86/VideoPipe) for video analysis.

[yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)，[demo0.mp4](https://www.ilanzou.com/s/yhUyq8f3)，[demo1.mp4](https://www.ilanzou.com/s/aIhyq8ET)

Please download the required `yolo11n.pt` model file and test video through the provided link, and save both to the `workspace` folder.

## Model Export

> [!IMPORTANT]
>
> Use the [`trtyolo-export`](https://github.com/laugh12321/TensorRT-YOLO/tree/export) tool package that comes with the project to export the ONNX model suitable for inference in this project and build it into a TensorRT engine.

Use the following command to export the ONNX format with the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin:

```bash
trtyolo export -w workspace/yolo11n.pt -v yolo11 -o workspace -b 2 -s
```

After running the above command, a `yolo11n.onnx` file with a `batch_size` of 2 will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine (fp16):

```bash
trtexec --onnx=workspace/yolo11n.onnx --saveEngine=workspace/yolo11n.engine --fp16
```

## Project Execution

1. Make sure that the project has been compiled according to the project documentation and the [`VideoPipe` compilation and debugging](https://github.com/sherlockchou86/VideoPipe/blob/master/README.md#compilation-and-debugging) (only the default five steps need to be executed, without adding any other compilation options).

2. Compile the project into an executable:

    ```bash
    cmake -S . -B build -D VIDEOPIPE_PATH="/path/to/your/VideoPipe"
    cmake --build . -j8 --config Release
    ```

    After compilation, the executable file will be generated in the `workspace` folder of the project root directory.

3. Run the following command for inference:

    ```bash
    cd workspace
    ./PipeDemo
    ```

<div align="center">
    <p>
        <img width="100%" src="../../assets/videopipe.png">
    </p>
</div>
