English | [简体中文](README.md)

# Model Inference Example

In this example, we demonstrate model inference using YOLOv8s model through CLI, Python, and C++.

## Model Export

Firstly, download the YOLOv8s model from [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) and save it to the `models` folder.

Then, export the model to ONNX format with the EfficientNMS plugin using the following command:

```bash
trtyolo export -w yolov8s.pt -v yolov8 -o models
```

After executing the above command, a file named `yolov8s.onnx` will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine:

```bash
trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s.engine --fp16
```

Next, we will perform model inference using different methods.

## Model Inference

Download the [coco128 dataset](https://ultralytics.com/assets/coco128.zip), unzip it, and move the images from the `coco128/images/train2017` folder to the `images` folder for inference.

### Inference Using CLI

You can perform inference using the `trtyolo` command-line tool provided by `tensorrt_yolo`. Run the following command to view help information related to inference:

```bash
trtyolo infer --help
```

Then, execute the following command for inference:

> For accelerated inference, use the --cudaGraph option, which supports only static models and not dynamic models.

```bash
trtyolo infer -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

This command will generate visual inference results in the `output` folder.

### Inference Using Python

You can also perform inference using the `tensorrt_yolo` library by writing scripts. `detect.py` is a pre-written script for inference.

> For accelerated inference, use the --cudaGraph option, which supports only static models and not dynamic models.

```bash
python detect.py -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

### Inference Using C++

Before performing inference using C++, ensure that you have compiled Deploy according to the [Deploy Compilation](../../docs/en/build_and_install.md#deploy-compilation).

Next, use xmake to compile `detect.cpp` into an executable:

```bash
xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy=/path/to/your/TensorRT-YOLO

xmake -P . -r
```

After executing the commands above, an executable named `detect` will be generated in the `build` directory at the root. Finally, you can run the executable directly or use `xmake run` for inference. Use `--help` to view detailed command options:

> For accelerated inference, use the --cudaGraph option, which supports only static models and not dynamic models.

```bash
xmake run -P . detect -e models/yolov8s.engine -i images -o output -l labels.txt --cudaGraph
```

These are the examples of performing model inference.
