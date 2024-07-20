English | [简体中文](README.md)

# Model Inference Examples

In this example, we demonstrate how to perform model inference using YOLOv8s with CLI, Python, and C++.

## Model Export

First, download the YOLOv8s model from [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) and save it to the `models` folder.

Then, export the model to ONNX format with the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin using the following command:

```bash
trtyolo export -w models/yolov8s.pt -v yolov8 -o models
```

After executing the above command, a file named `yolov8s.onnx` will be generated in the `models` folder. Next, convert the ONNX file to a TensorRT engine using the `trtexec` tool:

```bash
trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s.engine --fp16
```

Now, we will perform model inference using different methods.

## Model Inference

Download the [coco128](https://ultralytics.com/assets/coco128.zip) dataset, unzip it, and move the images from the `coco128/images/train2017` folder to the `images` folder for inference.

### Inference using CLI

You can use the `trtyolo` command-line tool provided by `tensorrt_yolo` for inference. Run the following command to see the help information related to inference:

```bash
trtyolo infer --help
```

Then, perform inference using the following command:

> To further speed up the inference process, use the `--cudaGraph` option, but this feature only supports static models, not dynamic models. (not supported before version 4.0)

```bash
trtyolo infer -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

This command will generate visualized inference results in the `output` folder.

### Inference using Python

You can also write a script using the `tensorrt_yolo` library for inference. The script `detect.py` is already written for this purpose.

> To further speed up the inference process, use the `--cudaGraph` option, but this feature only supports static models, not dynamic models.

```bash
python detect.py -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

### Inference using C++

Before performing inference using C++, ensure that you have compiled Deploy as per the [Deploy Build Guide](../../docs/en/build_and_install.md#deploy-Build).

Next, compile `detect.cpp` into an executable using xmake:

```bash
xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy=/path/to/your/TensorRT-YOLO

xmake -P . -r
```

After executing the above commands, an executable named `detect` will be generated in the `build` directory at the root. Finally, you can run the executable directly or use the `xmake run` command for inference. Use `--help` to see detailed command options:

> To further speed up the inference process, use the `--cudaGraph` option, but this feature only supports static models, not dynamic models.

```bash
xmake run -P . detect -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

The above are examples of how to perform model inference.