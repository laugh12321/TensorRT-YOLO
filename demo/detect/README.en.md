[简体中文](README.md) | English

# Model Inference Examples

This example demonstrates how to perform model inference using CLI, Python, and C++ with the YOLOv8s model as an example.

> [!IMPORTANT]  
> If you want to use the [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin) plugin to infer an OBB model, please refer to [Building TensorRT Custom Plugins](../../docs/en/build_trt_custom_plugin.md) for guidance.

## Model Export

### Detection Model

1. Download the [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt) model and save it to the `models` folder.
2. Use the following command to export the model to ONNX format with the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin:

    ```bash
    trtyolo export -w models/yolov8s.pt -v yolov8 -o models
    ```

    After running the above command, a `yolov8s.onnx` file will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine:

    ```bash
    trtexec --onnx=models/yolov8s.onnx --saveEngine=models/yolov8s.engine --fp16
    ```

### Oriented Bounding Box Model (OBB)

1. Download the [YOLOv8s-obb](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-obb.pt) model and save it to the `models` folder.
2. Use the following command to export the model to ONNX format with the [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin/) plugin:

    ```bash
    trtyolo export -w models/yolov8s-obb.pt -v yolov8 -o models
    ```

    After running the above command, a `yolov8s-obb.onnx` file will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine:

    ```bash
    trtexec --onnx=models/yolov8s-obb.onnx --saveEngine=models/yolov8s-obb.engine --fp16
    ```

## Dataset Preparation

### Detection Model

1. Download the [coco128](https://ultralytics.com/assets/coco128.zip) dataset.
2. After extraction, move the images from the `coco128/images/train2017` folder to the `images` folder for inference.

### Oriented Bounding Box Model (OBB)

1. Download the [DOTA-v1.0](https://drive.google.com/file/d/1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK/view) dataset.
2. After extraction, move the images from the `part1/images` folder to the `images` folder for inference.

## Model Inference

### Inference Using CLI

1. Use the `trtyolo` command-line tool for inference. Run the following command to view the help information:

    ```bash
    trtyolo infer --help
    ```

2. Run the following commands for inference:

    > [!NOTE] 
    > The `--cudaGraph` option, introduced in version 4.0, can further accelerate the inference process, but this feature only supports static models.
    > 
    > From version 4.2 onwards, OBB model inference is supported, with the new `-m, --mode` option for selecting Detection or OBB models.

    ```bash
    # Detection Model
    trtyolo infer -e models/yolov8s.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # Oriented Bounding Box Model
    trtyolo infer -e models/yolov8s-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

    The inference results will be saved to the `output` folder and generate visualized results.

### Inference Using Python

1. Use the `tensorrt_yolo` library for Python inference. The sample script `detect.py` is ready for use.
2. Run the following commands for inference:

    > [!NOTE] 
    > The `--cudaGraph` option can further accelerate the inference process, but this feature only supports static models.

    ```bash
    # Detection Model
    python detect.py -e models/yolov8s.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # Oriented Bounding Box Model
    python detect.py -e models/yolov8s-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

### Inference Using C++

1. Ensure that the project has been compiled according to the [Deploy Compilation Guide](../../docs/en/build_and_install.md#deploy-compilation).
2. Use `xmake` to compile `detect.cpp` into an executable file:

    ```bash
    xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy="/path/to/your/TensorRT-YOLO"

    xmake -P . -r
    ```

    After compilation, the executable file will be generated in the `build` folder at the project root.

3. Run the following commands for inference:

    > [!NOTE] 
    > The `--cudaGraph` option can further accelerate the inference process, but this feature only supports static models.

    ```bash
    # Detection Model
    xmake run -P . detect -e models/yolov8s.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # Oriented Bounding Box Model
    xmake run -P . detect -e models/yolov8s-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

By following the steps above, you can successfully complete model inference.
