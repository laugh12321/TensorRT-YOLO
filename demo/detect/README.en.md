[简体中文](README.md) | English

# Model Inference Example

This example uses the YOLO11n model to demonstrate how to perform model inference using CLI, Python, and C++.

## Model Export

> [!IMPORTANT]
>
> If you only want to export ONNX models (with TensorRT plugins) for inference in the `tensorrt_yolo` project through the command-line interface (CLI) tool `trtyolo`, you can install via [PyPI](https://pypi.org/project/tensorrt-yolo) by simply running the following command:
> 
> ```bash
> pip install -U tensorrt_yolo
> ```
> 
> If you want to experience the same inference speed as C++, please refer to [installing-tensorrt_yolo](../../docs/en/build_and_install.md#installing-tensorrt_yolo) to build the latest version of `tensorrt_yolo` yourself.

### Detection Model

1. Download the [YOLO11n](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) model and save it to the `models` folder.
2. Use the following command to export the ONNX format with [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin:

    ```bash
    trtyolo export -w models/yolo11n.pt -v yolo11 -o models
    ```

    After running the above command, a `yolo11n.onnx` file will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine:

    ```bash
    trtexec --onnx=models/yolo11n.onnx --saveEngine=models/yolo11n.engine --fp16
    ```

### Oriented Bounding Box Model (OBB)

1. Download the [YOLO11n-obb](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt) model and save it to the `models` folder.
2. Use the following command to export the ONNX format with [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin/) plugin:

    ```bash
    trtyolo export -w models/yolo11n-obb.pt -v yolo11 -o models
    ```

    After running the above command, a `yolo11n-obb.onnx` file will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine:

    ```bash
    trtexec --onnx=models/yolo11n-obb.onnx --saveEngine=models/yolo11n-obb.engine --fp16 --staticPlugins=/path/to/your/TensorRT-YOLO/lib/plugin/libcustom_plugins.so --setPluginsToSerialize=/path/to/your/TensorRT-YOLO/lib/plugin/libcustom_plugins.so
    ```

## Dataset Preparation

### Detection Model

1. Download the [coco128](https://ultralytics.com/assets/coco128.zip) dataset.
2. After extraction, move the images from the `coco128/images/train2017` folder to the `images` folder for inference.

### Oriented Bounding Box Model (OBB)

1. Download the [DOTA-v1.0](https://drive.google.com/file/d/1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK/view) dataset.
2. After extraction, move the images from the `part1/images` folder to the `images` folder for inference.

## Model Inference

> [!NOTE] 
> The `--cudaGraph` command added from version 4.0 can further accelerate the inference process, but this feature only supports static models.
> 
> From version 4.2, OBB model inference is supported, and the `-m, --mode` command is added to select between Detection and OBB models.

### Inference Using CLI

1. Use the `trtyolo` command-line tool for inference. Run the following command to view help information:

    ```bash
    trtyolo infer --help
    ```

2. Run the following command for inference:

    ```bash
    # Detection model
    trtyolo infer -e models/yolo11n.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # Oriented bounding box model
    trtyolo infer -e models/yolo11n-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

    Inference results will be saved to the `output` folder, with visualization results generated.

### Inference Using Python

1. Use the `tensorrt_yolo` library for Python inference. The example script `detect.py` is ready to use.
2. Run the following command for inference:

    ```bash
    # Detection model
    python detect.py -e models/yolo11n.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # Oriented bounding box model
    python detect.py -e models/yolo11n-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

### Inference Using C++

1. Ensure that the project has been compiled according to the [`TensorRT-YOLO` Compilation](../../docs/en/build_and_install.md#tensorrt-yolo-compilation).
2. Use xmake to compile `detect.cpp` into an executable:

    ```bash
    xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy="/path/to/your/TensorRT-YOLO"

    xmake -P . -r
    ```

    After compilation, the executable file will be generated in the `build` folder at the root of the project.

3. Run inference with the following command:

    ```bash
    # Detection model
    xmake run -P . detect -e models/yolo11n.engine -m 0 -i images -o output -l labels_det.txt --cudaGraph
    # Oriented bounding box model
    xmake run -P . detect -e models/yolo11n-obb.engine -m 1 -i images -o output -l labels_obb.txt --cudaGraph
    ```

> [!IMPORTANT]  
> When inferring with an OBB model built using the `--fp16` flag, there may be instances of duplicate anchor boxes. This issue is typically caused by a reduction in precision. Therefore, it is not recommended to build OBB models using the `--fp16` precision mode.

You can now successfully complete model inference using the above methods.
