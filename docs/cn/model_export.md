[English](../en/model_export.md) | 简体中文

# 模型导出

您可以使用 `tensorrt_yolo` 自带的 CLI 工具 trtyolo 导出 ONNX 模型，并使用 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件进行后处理。您可以使用命令 `trtyolo export --help` 查看具体的导出命令。

> 对于导出 PP-YOLOE 和 PP-YOLOE+ 的 ONNX 模型，仅会修改 `batch` 维度，`height` 和 `width` 维度不会更改。您需要在 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 中进行设置，默认为 `640`。
>
> [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800), [YOLOv7](https://github.com/WongKinYiu/yolov7#export), [YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) 官方仓库提供带 EfficientNMS 插件的 ONNX 模型导出， 这里不再二次提供。

```bash
# 导出使用远程仓库的 yolov3
trtyolo export -w yolov3.pt -v yolov3 -o output

# 导出使用本地仓库的 yolov5
trtyolo export -w yolov5s.pt -v yolov5 -o output --repo_dir your_local_yolovs_repository

# 使用 ultralytics 训练的 yolo 系列模型 (yolov3, yolov5, yolov6, yolov8, yolov9)，并指定 EfficientNMS 插件参数, 以动态 batch 导出
trtyolo export -w yolov8s.pt -v ultralytics -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# PP-YOLOE, PP-YOLOE+
trtyolo export --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output
```

导出的 ONNX 模型可以使用 `trtexec` 工具导出 TensorRT 模型。

```bash
# 静态 batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
# 动态 batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16
```
