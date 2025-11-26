[简体中文](README.md) | English

# Instance Segmentation Inference Example

This example uses the YOLO11n-seg model to demonstrate how to perform Instance Segmentation inference using the Command Line Interface (CLI), Python, and C++.

[yolo11n-seg.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt)，[【TestImages】COCO-part.zip](https://www.ilanzou.com/s/N5Oyq8hZ)

Please download the required `yolo11n-seg.pt` model file and test images through the provided link, and save the model file to the `models` folder, and place the extracted test images into the `images` folder after unzipping.

## Model Export

> [!IMPORTANT]
>
> Use the [`trtyolo-export`](https://github.com/laugh12321/TensorRT-YOLO/tree/export) tool package that comes with the project to export the ONNX model suitable for inference in this project and build it into a TensorRT engine.

Use the following command to export the ONNX format with the [EfficientIdxNMS](../../modules/plugin/efficientIdxNMSPlugin/) plugin:

```bash
trtyolo export -w models/yolo11n-seg.pt -v yolo11 -o models -s
```

After running the above command, a `yolo11n-seg.onnx` file with a `batch_size` of 1 will be generated in the `models` folder. Next, use the `trtexec` tool to convert the ONNX file to a TensorRT engine (fp16):

```bash
trtexec --onnx=models/yolo11n-seg.onnx --saveEngine=models/yolo11n-seg.engine --fp16 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so
```

## Model Inference

### Inference Using Python

1. Use the `trtyolo` library to run the example script `segment.py` for inference.
2. Run the following command for inference:

    ```bash
    python segment.py -e models/yolo11n-seg.engine -i images -o output -l labels.txt
    ```

### Inference Using C++

1. Ensure that the project has been compiled according to the project documentation.
2. Compile `segment.cpp` into an executable:

    ```bash
    cmake -S . -B build
    cmake --build build -j8 --config Release
    ```

    After compilation, the executable file will be generated in the `bin` folder of the project root directory.

3. Run the following command for inference:

    ```bash
    cd bin
    ./segment -e ../models/yolo11n-seg.engine -i ../images -o ../output -l ../labels.txt
    ```

Through the above methods, you can successfully complete model inference.
