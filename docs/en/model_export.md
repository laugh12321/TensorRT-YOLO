English | [中文](../cn/model_export.md)

# Exporting Models Using `trtyolo` CLI

The `tensorrt_yolo` package comes with a handy Command Line Interface (CLI) tool called `trtyolo` for exporting ONNX models suitable for inference with this project, complete with TensorRT plugins. To see the specific export commands, you can use `trtyolo export --help`.

> [!NOTE]  
> When exporting ONNX models for PP-YOLOE and PP-YOLOE+, only the `batch` dimension will be modified, while the `height` and `width` dimensions will remain unchanged. You will need to set this in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), with a default value typically being `640`.
>
> Official repositories such as [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export), [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) already provide ONNX model exports with the EfficientNMS plugin, so they are not duplicated here.
>

### Export Command Examples
```bash
# Export YOLOv3 model from a remote repository
trtyolo export -w yolov3.pt -v yolov3 -o output

# Export YOLOv5 model from a local repository
trtyolo export -w yolov5s.pt -v yolov5 -o output --repo_dir your_local_yolovs_repository

# Export Ultralytics-trained yolo series models (YOLOv3, YOLOv5, YOLOv6, YOLOv8, YOLOv9, YOLOv10, YOLO11) with plugin parameters for dynamic batch export
trtyolo export -w yolov8s.pt -v ultralytics -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# Export YOLOv10 model
trtyolo export -w yolov10s.pt -v yolov10 -o output

# Export YOLO11 OBB model
trtyolo export -w yolov11n-obb.pt -v yolo11 -o output

# Export PP-YOLOE, PP-YOLOE+ models
trtyolo export --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output
```

### Using `trtexec` to Export TensorRT Models
The exported ONNX models can be further exported to TensorRT models using the `trtexec` tool.

```bash
# Static batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# Dynamic batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

# YOLOv8-OBB static batch
trtexec --onnx=yolov8n-obb.pt --saveEngine=yolov8n-obb.engine --fp16 --staticPlugins=./lib/plugin/libcustom_plugins.so --setPluginsToSerialize=./lib/plugin/libcustom_plugins.so

# YOLO11-OBB dynamic batch
trtexec --onnx=yolov11n-obb.pt --saveEngine=yolov11n-obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --dynamicPlugins=./lib/plugin/custom_plugins.dll --setPluginsToSerialize=./lib/plugin/custom_plugins.dll
```

> [!NOTE]  
> When using the `--dynamicPlugins` and `--setPluginsToSerialize` parameters to build a dynamic model with custom plugins, encountering the error `[E] Error[4]: IRuntime::deserializeCudaEngine: Error Code 4: API Usage Error (Cannot register the library as plugin creator of EfficientRotatedNMS_TRT exists already.)` typically indicates that the engine has been successfully built, but an attempt to load and deserialize the engine detected a duplicate plugin registration. This error can be safely ignored.
