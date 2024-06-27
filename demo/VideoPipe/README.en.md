English | [简体中文](README.md)

# Video Analysis Example

This example uses the YOLOv8s model to demonstrate how to integrate the TensorRT-YOLO Deploy module into [VideoPipe](https://github.com/sherlockchou86/VideoPipe) for video analysis.

## Model Export

First, download the YOLOv8s model from [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) and save it to the `workspace` folder.

Next, use the following command to export the model to ONNX format with the EfficientNMS plugin from [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin):

```bash
cd workspace
trtyolo export -w yolov8s.pt -v yolov8 -o models -b 2
```

After executing the command above, a file named `yolov8s.onnx` will be generated in the `models` folder. Then, convert the ONNX file to a TensorRT engine using the `trtexec` tool:

```bash
cd workspace
trtexec --onnx=yolov8s.onnx --saveEngine=yolov8s.engine --fp16
```

## Project Execution

Before performing inference, make sure VideoPipe and TensorRT-YOLO have been compiled.

Next, use xmake to compile the project into an executable:

```bash
xmake f -P . --tensorrt=/path/to/your/TensorRT --deploy=/path/to/your/TensorRT-YOLO --videopipe=/path/to/your/VideoPipe

xmake -P . -r
```

After successful compilation, you can directly run the generated executable or use the `xmake run` command for inference:

```bash
xmake run -P . PipeDemo
```

<div align="center">
    <p>
        <img width="100%" src="../../assets/videopipe.jpg">
    </p>
</div>

The above demonstrates the method for performing model inference.
