English | [中文](../cn/model_export.md)

# Model Export

## Exporting Models Using `trtyolo` CLI

`tensorrt_yolo` provides a convenient Command Line Interface (CLI) tool, `trtyolo`, for exporting ONNX models into formats suitable for inference in this project, with integrated TensorRT plugins. You can view detailed export command instructions by running `trtyolo export --help`.

> [!NOTE]  
> When exporting ONNX models for PP-YOLOE and PP-YOLOE+, only the batch dimension is adjusted, while the height and width dimensions remain unchanged. You need to configure this in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), with the default value typically set to 640.
>
> Official repositories such as [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export), and [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) already provide ONNX model exports with the EfficientNMS plugin, so this functionality is not duplicated here.

### Export Command Examples

```bash
# Export a YOLOv3 model from a remote repository
trtyolo export -w yolov3.pt -v yolov3 -o output

# Export a YOLOv5 model from a local repository
trtyolo export -w yolov5s.pt -v yolov5 -o output --repo_dir your_local_yolovs_repository

# Export models trained with Ultralytics (YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv9, YOLOv10, YOLO11) with plugin parameters, using dynamic batch export
trtyolo export -w yolov8s.pt -v ultralytics -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# Export a YOLOv10 model with height 1080 and width 1920
trtyolo export -w yolov10s.pt -v yolov10 -o output --imgsz 1080 1920

# Export a YOLO11 OBB model
trtyolo export -w yolo11n-obb.pt -v yolo11 -o output

# Export PP-YOLOE and PP-YOLOE+ models
trtyolo export --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output
```

## Building TensorRT Engines Using `trtexec`

Exported ONNX models can be built into TensorRT engines using the `trtexec` tool.

```bash
# Static batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# Dynamic batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

# YOLOv8-OBB static batch
trtexec --onnx=yolov8n-obb.onnx --saveEngine=yolov8n-obb.engine --fp16 --staticPlugins=./lib/plugin/libcustom_plugins.so --setPluginsToSerialize=./lib/plugin/libcustom_plugins.so

# YOLO11-OBB dynamic batch
trtexec --onnx=yolo11n-obb.onnx --saveEngine=yolo11n-obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --staticPlugins=./lib/plugin/custom_plugins.dll --setPluginsToSerialize=./lib/plugin/custom_plugins.dll
```

> [!NOTE]  
> When building dynamic models with custom plugins using the `--staticPlugins` and `--setPluginsToSerialize` parameters, if you encounter the error `[E] Error[4]: IRuntime::deserializeCudaEngine: Error Code 4: API Usage Error (Cannot register the library as plugin creator of EfficientRotatedNMS_TRT exists already.)`, this typically indicates that the engine build was successful, but a plugin duplicate registration was detected during engine loading and deserialization. In such cases, this error can be safely ignored.
