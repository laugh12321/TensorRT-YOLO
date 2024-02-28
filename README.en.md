## <div align="center">🚀 TensorRT YOLO</div>

TensorRT-YOLO is an inference acceleration project that supports YOLOv5, YOLOv8, PP-YOLOE, and PP-YOLOE+ using NVIDIA TensorRT for optimization. The project integrates EfficientNMS TensorRT plugin for enhanced post-processing and utilizes CUDA kernel functions to accelerate the preprocessing phase. TensorRT-YOLO provides support for both C++ and Python inference, aiming to deliver a fast and optimized object detection solution.

## <div align="center">✨ Key Features</div>

- Supports FLOAT32, FLOAT16 ONNX export, and TensorRT inference
- Supports YOLOv5, YOLOv8, PP-YOLOE, and PP-YOLOE+
- Integrates EfficientNMS TensorRT plugin for accelerated post-processing
- Utilizes CUDA kernel functions to accelerate preprocessing
- Supports C++ and Python inference

## <div align="center">🛠️ Requirements</div>

- Recommended CUDA version >= 11.4
- Recommended TensorRT version >= 8.4

## <div align="center">📦 Usage Guide</div>

<details open>
<summary>Installation</summary>

Clone the repo and install the dependencies from [**Python>=3.8.0**](https://www.python.org/) using [requirements.txt](https://github.com/laugh12321/TensorRT-YOLO/blob/master/requirements.txt). Ensure [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/) (for YOLOv5 and YOLOv8 export) and [**PaddlePaddle>=2.5**](https://www.paddlepaddle.org.cn/install/quick/) (for PP-YOLOE and PP-YOLOE+ export) are installed.

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  # clone
cd TensorRT-YOLO
pip install -r requirements.txt  # install
pip install ultralytics          # Optional, export YOLOv5 and YOLOv8
pip install paddle2onnx          # Optional, export PP-YOLOE and PP-YOLOE+
```
</details>

<details>
<summary>Model Export</summary>

Use the following commands to export ONNX models and add the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin for post-processing.

**Note:** For exporting ONNX models of PP-YOLOE and PP-YOLOE+, the input image size `imgsz` must match the size exported by [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), which is the default `640`.

**YOLOv5**
```bash
python python/export/yolov5/export.py -w yolov5s.pt -o output -b 8 --img 640 -s --half
```

**YOLOv8**
```bash
python python/export/yolov8/export.py -w yolov8s.pt -o output --conf-thres 0.25 --iou-thres 0.45 --max-boxes 100
```

**PP-YOLOE and PP-YOLOE+**
```bash
python python/export/ppyoloe/export.py --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output
```

Exported ONNX models are then exported to TensorRT models using the `trtexec` tool.

**Note:** ONNX models exported with `python export.py --half` must include `--fp16` when using `trtexec`.

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```
</details>

<details>
<summary>Inference using detect.py</summary>

`detect.py` currently supports inference on a single image or batch inference on an entire directory. You can specify the inference data using the `--inputs` parameter. The results of the inference can be saved to a specified path using the `--output` parameter, with the default being `None` indicating no saving. For detailed command descriptions, please run `python detect.py -h`.

```bash
python detect.py  -e model.engine -o output -i img.jpg                         # image
                                               path/                           # directory
```
</details>

## <div align="center">📄 License</div>

TensorRT-YOLO is licensed under the **GPL-3.0 License**, an [OSI-approved](https://opensource.org/licenses/) open-source license that is ideal for students and enthusiasts, fostering open collaboration and knowledge sharing. Please refer to the [LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) file for more details.

Thank you for choosing TensorRT-YOLO; we encourage open collaboration and knowledge sharing, and we hope you comply with the relevant provisions of the open-source license.

## <div align="center">📞 Contact</div>

For bug reports and feature requests regarding TensorRT-YOLO, please visit [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)!
