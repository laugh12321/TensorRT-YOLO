English | [中文](../cn/model_export.md)

# Model Export

## Exporting Models Using `trtyolo` CLI

`tensorrt_yolo` provides a convenient Command Line Interface (CLI) tool, `trtyolo`, for exporting ONNX models into formats suitable for inference in this project, with integrated TensorRT plugins. You can view detailed export command instructions by running `trtyolo export --help`.

### Parameter Descriptions
- `-v, --version`: Model version. Options include `yolov3`, `yolov5`, `yolov8`, `yolov10`, `yolo11`, `yolo12`, `yolo-world`, `yoloe`, `pp-yoloe`, `ultralytics`.
- `-o, --output`: Directory path to save the exported model.
- `-w, --weights`: Path to PyTorch YOLO weights (required for non PP-YOLOE models).
- `--model_dir`: Directory path containing the PaddleDetection PP-YOLOE model (required for PP-YOLOE).
- `--model_filename`: Filename of the PaddleDetection PP-YOLOE model (required for PP-YOLOE).
- `--params_filename`: Filename of the PaddleDetection PP-YOLOE parameters (required for PP-YOLOE).
- `-b, --batch`: Total batch size for the model. Use `-1` for dynamic batch size. Defaults to `1`.
- `--max_boxes`: Maximum number of detections per image. Defaults to `100`.
- `--iou_thres`: NMS IoU threshold for post-processing. Defaults to `0.45`.
- `--conf_thres`: Confidence threshold for object detection. Defaults to `0.25`.
- `--imgsz`: Image size (single value for square or "height,width"). Defaults to `640` (for non PP-YOLOE models).
- `--names`: Custom class names for YOLO-World and YOLOE (comma-separated, e.g., "person,car,dog"). Only applicable for YOLO-World and YOLOE models.
- `--repo_dir`: Directory containing the local repository (if using `torch.hub.load`). Only applicable for YOLOv3 and YOLOv5 models.
- `--opset`: ONNX opset version. Defaults to `12`.
- `-s, --simplify`: Whether to simplify the exported ONNX model. Defaults to `False`.

> [!NOTE]
> When exporting ONNX models for PP-YOLOE and PP-YOLOE+, only the batch dimension is adjusted, while the height and width dimensions remain unchanged. You need to configure this in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), with the default value typically set to 640.
>
> Official repositories such as [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export), and [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) already provide ONNX model exports with the EfficientNMS plugin, so this functionality is not duplicated here.

### Export Command Examples

```bash
# Exporting a YOLOv3 model from a remote repository
trtyolo export -v yolov3 -w yolov3.pt -o output

# Exporting a YOLOv5 Classify model from a local repository
trtyolo export -v yolov5 -w yolov5s-cls.pt -o output --repo_dir your_local_yolovs_repository

# Exporting Ultralytics-trained YOLO series models (YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv9, YOLOv10, YOLO11, etc.) with plugin parameters for dynamic batch export
trtyolo export -v ultralytics -w yolov8s.pt -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# Exporting PP-YOLOE and PP-YOLOE+ models
trtyolo export -v pp-yoloe --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output

# Exporting a YOLOv10 model with height 1080 and width 1920
trtyolo export -v yolov10 -w yolov10s.pt -o output --imgsz 1080,1920

# Exporting a YOLO11 OBB model
trtyolo export -v yolo11 -w yolo11n-obb.pt -o output

# Exporting a YOLO12 Segment model
trtyolo export -v yolo12 -w yolo12n-seg.pt -o output

# Exporting a YOLO-World model with custom classes
trtyolo export -v yolo-world -w yoloworld.pt -o output --names "person,car,dog"

# Exporting a YOLOE model
trtyolo export -v yoloe -w yoloe.pt -o output
```

## Building TensorRT Engines Using `trtexec`

Exported ONNX models can be built into TensorRT engines using the `trtexec` tool.

```bash
# Static batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# Dynamic batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

# YOLOv8-OBB static batch
trtexec --onnx=yolov8n-obb.onnx --saveEngine=yolov8n-obb.engine --fp16 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/libcustom_plugins.so

# YOLO11-OBB dynamic batch
trtexec --onnx=yolo11n-obb.onnx --saveEngine=yolo11n-obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --staticPlugins=/your/tensorrt-yolo/install/dir/lib/custom_plugins.dll --setPluginsToSerialize=/your/tensorrt-yolo/install/dir/lib/custom_plugins.dll
```

> [!NOTE]
> When building dynamic models with custom plugins using the `--staticPlugins` and `--setPluginsToSerialize` parameters, if you encounter the error `[E] Error[4]: IRuntime::deserializeCudaEngine: Error Code 4: API Usage Error (Cannot register the library as plugin creator of EfficientRotatedNMS_TRT exists already.)`, this typically indicates that the engine build was successful, but a plugin duplicate registration was detected during engine loading and deserialization. In such cases, this error can be safely ignored.
