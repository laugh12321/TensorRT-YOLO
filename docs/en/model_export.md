English | [中文](../cn/model_export.md)

# Model Export

You can use the `trtyolo` CLI tool provided by `tensorrt_yolo` to export ONNX models and perform post-processing with the [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) plugin. You can check the specific export command using `trtyolo export --help`.

**Note:** When exporting ONNX models for PP-YOLOE and PP-YOLOE+, only the `batch` dimension will be modified, while the `height` and `width` dimensions will remain unchanged. You need to set this in [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection), defaulting to `640`.

```bash
# Use local repository for yolov5
trtyolo export -w yolov5s.pt -v yolov5 -o output --repo_dir your_local_yolovs_repository
# yolov8
trtyolo export -w yolov8s.pt -v yolov8 -o output
# yolov9 dynamic batch using github repository
trtyolo export -w yolov9-c.pt -v yolov9 -b -1 -o output
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
