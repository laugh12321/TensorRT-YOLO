[简体中文](README.md) | English

# Image Classification Inference Example

This example uses the yolo11n-cls model to demonstrate how to perform Image Classification inference using the Command Line Interface (CLI), Python, and C++.

[yolo11n-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-cls.pt)，[【TestImages】ImageNet-part.zip](https://www.ilanzou.com/s/1UsyPhED)

Please download the required `yolo11n-cls.pt` model file and test images through the provided link, and save the model file to the `models` folder, and place the extracted test images into the `images` folder after unzipping.

## Model Convert

> [!IMPORTANT]
>
> Please first export the model weights to ONNX, then use the bundled [`trtyolo-export`](https://github.com/laugh12321/trtyolo-export) tool to convert the ONNX model into TensorRT-YOLO compatible outputs and build it into a TensorRT engine.

Use the following commands to export ONNX first and then convert it into the structure required by this project:

```bash
yolo export model=models/yolo11n-cls.pt format=onnx imgsz=224 batch=1
trtyolo-export -i models/yolo11n-cls.onnx -o models/yolo11n-cls-trtyolo.onnx -s
```

After running the commands above, the `models` folder will contain the original ONNX file `yolo11n-cls.onnx` and the converted file `yolo11n-cls-trtyolo.onnx`. Next, use `trtexec` to build a TensorRT engine from the converted ONNX file (fp16):

```bash
trtexec --onnx=models/yolo11n-cls-trtyolo.onnx --saveEngine=models/yolo11n-cls.engine --fp16
```

## Model Inference

### Inference Using Python

1. Use the `trtyolo` library to run the example script `classify.py` for inference.
2. Run the following command for inference:

    ```bash
    python classify.py -e models/yolo11n-cls.engine -i images -o output -l labels.txt
    ```

### Inference Using C++

1. Ensure that the project has been compiled according to the documentation.
2. Compile `classify.cpp` into an executable:

    ```bash
    cmake -S . -B build
    cmake --build build -j8 --config Release
    ```

    After compilation, the executable file will be generated in the `bin` folder of the project root directory.

3. Run the following command for inference:

    ```bash
    cd bin
    ./classify -e ../models/yolo11n-cls.engine -i ../images -o ../output -l ../labels.txt
    ```

Through the above methods, you can successfully complete model inference.
