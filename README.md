# TensorRT-YOLO

Accelerated YOLOv5, YOLOv8, PP-YOLOE, PP-YOLOE+ inference with NVIDIA TensorRT. Fast, optimized, and GPU-accelerated object detection. Includes EfficientNMS TensorRT plugin for enhanced post-processing and supports both C++ and Python inference.


## Requirements

- CUDA >= 11.7
- TensorRT >= 8.4

Install python requirements.

```bash
pip install -r requirements.txt
```

## Export ONNX model with NMS and convert to TensorRT engine.

### YOLOv5

Install additional Python packages:

```bash
pip install torch torchvision
pip install "git+https://github.com/zhiqwang/yolort.git"
```

Run the export script:

```bash
python python/export/yolov5/export.py \
-w yolov5s.pt \
-o output \
-b 4 \
--img 640 \
--conf-thres 0.25 \
--iou-thres 0.45 \
--max-boxes 100 \
-p fp16 \
-s \
--verbose \
--workspace 4
```

### YOLOv8

Install additional Python packages:

```bash
pip install torch torchvision
pip install ultralytics
```

Run the export script:

```bash
python python/export/yolov8/export.py \
-w yolov8s.pt \
-o output \
-b 4 \
--img 640 \
--conf-thres 0.25 \
--iou-thres 0.45 \
--max-boxes 100 \
-p fp16 \
-s \
--verbose \
--workspace 4 \
--opset 11
```

### PP-YOLOE

Install additional Python packages (If using paddle2onnx version `1.0.6`, make sure the paddlepaddle version is `2.5.2`) :

```bash
pip install paddlepaddle-gpu==2.5.2.post117 -f https://www.paddlepaddle.org.cn/whl/windows/mkl/avx/stable.html
pip install paddle2onnx
```

Run the export script:

```bash
python python/export/ppyoloe/export.py \
--model_dir ppyoloe_crn_m_300e_coco \
--model_filename model.pdmodel \
--params_filename model.pdiparams \
-o output \
-b 4 \
--img 640 \
--conf-thres 0.25 \
--iou-thres 0.45 \
--max-boxes 100 \
-p fp16 \
-s \
--verbose \
--workspace 4 \
--opset 11
```

## Inference

### Inference with Python

Use the `detect.py` script to perform image or directory-based inference in YOLO Series. When the source is a directory, benchmarks can be run.

```bash
python detect.py  \
-w model.engine \
-s path \   # or img.jpg
-o output \
--max-image-size 1080 1920 \
--benchmark # when the source is a directory
```

### Inference with C++