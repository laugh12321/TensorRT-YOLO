English | [中文](../cn/model_export.md)

# Model Export

You can use the `trtyolo` CLI tool provided by `tensorrt_yolo` to export ONNX models and perform post-processing with the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin. You can check the specific export command using `trtyolo export --help`.

> When exporting ONNX models for PP-YOLOE and PP-YOLOE+, only the `batch` dimension will be modified, while the `height` and `width` dimensions will remain unchanged. You need to set this in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), defaulting to `640`.
>
> The official repositories for [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export) and [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) provide ONNX model exports with the EfficientNMS plugin included. Therefore, there is no need to re-provide these exports here.

```bash
# Use remote repository for yolov5
trtyolo export -w yolov3.pt -v yolov3 -o output

# Use local repository for yolov5
trtyolo export -w yolov5s.pt -v yolov5 -o output --repo_dir your_local_yolovs_repository

# Use ultralytics trained YOLO series models (yolov3, yolov5, yolov6, yolov8, yolov9), and specify EfficientNMS plugin parameters with dynamic batch
trtyolo export -w yolov8s.pt -v ultralytics -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# PP-YOLOE, PP-YOLOE+
trtyolo export --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output
```

The generated ONNX models can be exported as TensorRT models using the `trtexec` tool.

```bash
# Static
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
# Dynamic
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16
```
