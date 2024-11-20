[简体中文](README.md) | English

# Pose Estimation Inference Example

This example uses the yolo11n-pose model to demonstrate how to perform pose estimation inference using the Command Line Interface (CLI), Python, and C++.

[yolo11n-pose.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt)，[【TestImages】COCO-Pose-part.zip](https://www.ilanzou.com/s/kBby4w1D)

Please download the required `yolo11n-pose.pt` model file and test images through the provided link, and save the model file to the `models` folder, and place the extracted test images into the `images` folder after unzipping.

## Model Export

> [!IMPORTANT]
>
> If you only want to export the ONNX model (with TensorRT plugins) that can be used for inference in this project through the `tensorrt_yolo` provided Command Line Interface (CLI) tool `trtyolo`, you can install it via [PyPI](https://pypi.org/project/tensorrt-yolo) by simply executing the following command:
>
> ```bash
> pip install -U tensorrt_yolo
> ```
> 
> If you want to experience the same inference speed as C++, please refer to [Install-tensorrt_yolo](../../docs/en/build_and_install.md#install-tensorrt_yolo) to build the latest version of `tensorrt_yolo` yourself.

Use the following command to export the ONNX format with the [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin/) plugin. For detailed `trtyolo` CLI export methods, please read [Model Export](../../docs/en/model_export.md):

```bash
trtyolo export -w models/yolo11n-pose.pt -v yolo11 -o models -s
```

After running the above command, a `yolo11n-pose.onnx` file with a `batch_size` of 1 will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine (fp16):

```bash
trtexec --onnx=models/yolo11n-pose.onnx --saveEngine=models/yolo11n-pose.engine --fp16 --staticPlugins=/path/to/your/TensorRT-YOLO/lib/plugin/libcustom_plugins.so --setPluginsToSerialize=/path/to/your/TensorRT-YOLO/lib/plugin/libcustom_plugins.so
```

## Model Inference

> [!IMPORTANT]
>
> The `tensorrt_yolo` installed via [PyPI](https://pypi.org/project/tensorrt-yolo) only provides the ONNX model (with TensorRT plugins) for inference in this project and does not provide inference capabilities.
> If you want to experience the same inference speed as C++, please refer to [Install-tensorrt_yolo](../../docs/en/build_and_install.md#install-tensorrt_yolo) to build the latest version of `tensorrt_yolo` yourself.

### Inference Using CLI

> [!NOTE] 
> The `--cudaGraph` command added from version 4.0 can further accelerate the inference process, but this feature only supports static models.
> 
> From version 4.3 and later, support for pose estimation inference is added. The command `-m 2, --mode 2` is used to select the pose estimation.

1. Use the `trtyolo` command-line tool from the `tensorrt_yolo` library for inference. Run the following command to view help information:

    ```bash
    trtyolo infer --help
    ```

2. Run the following command for inference:

    ```bash
    trtyolo infer -e models/yolo11n-pose.engine -m 3 -i images -o output -l labels.txt --cudaGraph
    ```

    The inference results will be saved in the `output` folder, and a visualization result will be generated.

### Inference Using Python

1. Use the `tensorrt_yolo` library to run the example script `pose.py` for inference.
2. Run the following command for inference:

    ```bash
    python pose.py -e models/yolo11n-pose.engine -i images -o output -l labels.txt --cudaGraph
    ```

### Inference Using C++

1. Ensure that the project has been compiled according to the [`TensorRT-YOLO` Compilation](../../docs/en/build_and_install.md#tensorrt-yolo-compile).
2. Compile `pose.cpp` into an executable:

    ```bash
    # Compile using xmake
    xmake f -P . --tensorrt="/path/to/your/TensorRT" --deploy="/path/to/your/TensorRT-YOLO"
    xmake -P . -r

    # Compile using cmake
    mkdir -p build && cd build
    cmake -DTENSORRT_PATH="/path/to/your/TensorRT" -DDEPLOY_PATH="/path/to/your/TensorRT-YOLO" .. 
    cmake --build . -j8 --config Release
    ```

    After compilation, the executable file will be generated in the `bin` folder of the project root directory.

3. Run the following command for inference:

    ```bash
    cd bin
    ./pose -e ../models/yolo11n-pose.engine -i ../images -o ../output -l ../labels.txt --cudaGraph
    ```

Through the above methods, you can successfully complete model inference.
