English | [简体中文](README.md)

## <div align="center">🚀 TensorRT YOLO</div>

<p align="center">
    <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/commits"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/laugh12321/TensorRT-YOLO?style=for-the-badge&color=rgb(47%2C154%2C231)"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2350e472">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2320878f">
</p>

TensorRT-YOLO is an inference acceleration project that supports YOLOv5, YOLOv8, YOLOv9, PP-YOLOE, and PP-YOLOE+ using NVIDIA TensorRT for optimization. The project integrates EfficientNMS TensorRT plugin for enhanced post-processing and utilizes CUDA kernel functions to accelerate the preprocessing phase. TensorRT-YOLO provides support for both C++ and Python inference, aiming to deliver a fast and optimized object detection solution.

<div align="center">
    <img src=assets/example0.jpg height="320">
    <img src=assets/example1.jpg height="320">
</div>

## <div align="center">✨ Key Features</div>

- Supports YOLOv5, YOLOv8, YOLOv9, PP-YOLOE, and PP-YOLOE+
- Supports static and dynamic export to ONNX, as well as TensorRT inference
- Integrated EfficientNMS TensorRT plugin for accelerated post-processing
- Utilizes CUDA kernel functions for accelerated pre-processing
- Supports inference in both C++ and Python

## <div align="center">🛠️ Requirements</div>

- Recommended CUDA version >= 11.7
- Recommended TensorRT version >= 8.6

## <div align="center">📦 Usage Guide</div>

<details open>
<summary>Installation</summary>

Clone the repo and install the dependencies from [**Python>=3.8.0**](https://www.python.org/) using [requirements.txt](https://github.com/laugh12321/TensorRT-YOLO/blob/master/requirements.txt). Ensure [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) (for YOLOv5, YOLOv8 and YOLOv9 export) and [**PaddlePaddle>=2.5**](https://www.paddlepaddle.org.cn/install/quick/) (for PP-YOLOE and PP-YOLOE+ export) are installed.

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  # clone
cd TensorRT-YOLO
pip install -r requirements.txt  # install
pip install ultralytics          # Optional, export YOLOv5, YOLOv8 and YOLOv9
pip install paddle2onnx          # Optional, export PP-YOLOE and PP-YOLOE+
```
</details>

<details>
<summary>Model Export</summary>

Use the following commands to export ONNX models and add the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin for post-processing.

**Note:** Exporting ONNX models for PP-YOLOE and PP-YOLOE+ only modifies the `batch` dimension, and the `height` and `width` dimensions cannot be changed. These dimensions are set in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) and default at `640`.

**YOLOv5, v8, v9**

```bash
# Static
python python/export/{yolo version}/export.py -w your_model_path.pt -o output -b 8 --img 640 -s
# Dynamic
python python/export/{yolo version}/export.py -w your_model_path.pt -o output -s --dynamic
```

**PP-YOLOE and PP-YOLOE+**

```bash
# Static
python python/export/ppyoloe/export.py --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output -b 8 -s
# Dynamic
python python/export/ppyoloe/export.py --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output -s --dynamic
```

Exported ONNX models are then exported to TensorRT models using the `trtexec` tool.


```bash
# Static
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
# Dynamic
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16
```

</details>

<details>
<summary>Inference using detect.py</summary>

`detect.py` currently supports inference on a single image or batch inference on an entire directory. You can specify the inference data using the `--inputs` parameter. The results of the inference can be saved to a specified path using the `--output` parameter, with the default being `None` indicating no saving. For detailed command descriptions, please run `python detect.py -h`.

```bash
python detect.py -e model.engine -o output -i img.jpg                         # image
                                               path/                           # directory
```
</details>

<details>
<summary>Inference using detect.cpp</summary>

The commands for `detect.cpp` are consistent with those for `detect.py`. Here we use `xmake` for compilation.

```bash
detect -e model.engine -o output -i img.jpg                         # image
                                     path/                           # directory
```
</details>

## <div align="center">📄 License</div>

TensorRT-YOLO is licensed under the **GPL-3.0 License**, an [OSI-approved](https://opensource.org/licenses/) open-source license that is ideal for students and enthusiasts, fostering open collaboration and knowledge sharing. Please refer to the [LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) file for more details.

Thank you for choosing TensorRT-YOLO; we encourage open collaboration and knowledge sharing, and we hope you comply with the relevant provisions of the open-source license.

## <div align="center">📞 Contact</div>

For bug reports and feature requests regarding TensorRT-YOLO, please visit [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)!
