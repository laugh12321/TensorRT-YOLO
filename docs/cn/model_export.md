[English](../en/model_export.md) | 简体中文

# 模型导出

## 使用 `trtyolo` CLI 导出模型

`tensorrt_yolo` 提供了一个便捷的命令行界面（CLI）工具 `trtyolo`，用于将模型导出为适用于项目推理的格式，并集成了 TensorRT 插件。您可以通过运行 `trtyolo export --help` 查看详细的导出命令帮助信息。

### 参数说明
- `-v, --version`: 模型版本。支持的选项包括 `yolov3`, `yolov5`, `yolov8`, `yolov10`, `yolo11`, `yolo12`, `yolo-world`, `yoloe`, `pp-yoloe`, `ultralytics`。
- `-o, --output`: 导出模型保存的目录路径。
- `-w, --weights`: PyTorch YOLO 权重文件路径（非 PP-YOLOE 模型必需）。
- `--model_dir`: 包含 PaddleDetection PP-YOLOE 模型的目录路径（PP-YOLOE 模型必需）。
- `--model_filename`: PaddleDetection PP-YOLOE 模型文件名（PP-YOLOE 模型必需）。
- `--params_filename`: PaddleDetection PP-YOLOE 参数文件名（PP-YOLOE 模型必需）。
- `-b, --batch`: 模型的总批量大小。使用 `-1` 表示动态批量大小，默认为 `1`。
- `--max_boxes`: 每张图像的最大检测框数量，默认为 `100`。
- `--iou_thres`: NMS 的 IoU 阈值，默认为 `0.45`。
- `--conf_thres`: 目标检测的置信度阈值，默认为 `0.25`。
- `--imgsz`: 图像大小（单个值表示正方形，或 "height,width"）。默认为 `640`（非 PP-YOLOE 模型）。
- `--names`: 自定义类别名称（仅适用于 YOLO-World 和 YOLOE 模型，以逗号分隔，例如 "person,car,dog"）。
- `--repo_dir`: 包含本地仓库的目录路径（仅适用于 YOLOv3 和 YOLOv5 模型，使用 `torch.hub.load` 时适用）。
- `--opset`: ONNX opset 版本，默认为 `12`。
- `-s, --simplify`: 是否简化导出的 ONNX 模型，默认为 `False`。

> [!NOTE]
> 在导出 PP-YOLOE 和 PP-YOLOE+ 的 ONNX 模型时，仅会调整 batch 维度，而 height 和 width 维度保持不变。您需要在 [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection) 中进行相关设置，默认值通常为 640。
>
> 官方仓库如 [YOLOv6](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800)、[YOLOv7](https://github.com/WongKinYiu/yolov7#export)、[YOLOv9](https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461) 已经提供了带有 EfficientNMS 插件的 ONNX 模型导出功能，因此此处不再重复提供。

### 导出命令示例

```bash
# 导出远程仓库中的 YOLOv3 模型
trtyolo export -v yolov3 -w yolov3.pt -o output

# 导出本地仓库中的 YOLOv5 Classify 模型
trtyolo export -v yolov5 -w yolov5s-cls.pt -o output --repo_dir your_local_yolovs_repository

# 使用 Ultralytics 训练的 YOLO 系列模型（YOLOv3、YOLOv5、YOLOv6、YOLOv8、YOLOv9、YOLOv10、YOLO11 等），并指定插件参数，以动态 batch 导出
trtyolo export -v ultralytics -w yolov8s.pt -o output --max_boxes 100 --iou_thres 0.45 --conf_thres 0.25 -b -1

# 导出 PP-YOLOE 和 PP-YOLOE+ 模型
trtyolo export -v pp-yoloe --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output

# 导出 YOLOv10 模型，高度 1080，宽度 1920
trtyolo export -v yolov10 -w yolov10s.pt -o output --imgsz 1080,1920

# 导出 YOLO11 OBB 模型
trtyolo export -v yolo11 -w yolo11n-obb.pt -o output

# 导出 YOLO12 Segment 模型
trtyolo export -v yolo12 -w yolo12n-seg.pt -o output

# 导出 YOLO-World 模型，并自定义类别
trtyolo export -v yolo-world -w yoloworld.pt -o output --names "person,car,dog"

# 导出 YOLOE 模型
trtyolo export -v yoloe -w yoloe.pt -o output
```

## 使用 `trtexec` 构建 TensorRT 引擎

导出的 ONNX 模型可以通过 `trtexec` 工具构建为 TensorRT 引擎。

```bash
# 静态 batch
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16

# 动态 batch
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16

# YOLOv8-OBB 静态 batch
trtexec --onnx=yolov8n-obb.onnx --saveEngine=yolov8n-obb.engine --fp16 --staticPlugins=./lib/plugin/libcustom_plugins.so --setPluginsToSerialize=./lib/plugin/libcustom_plugins.so

# YOLO11-OBB 动态 batch
trtexec --onnx=yolo11n-obb.onnx --saveEngine=yolo11n-obb.engine --fp16 --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --staticPlugins=./lib/plugin/custom_plugins.dll --setPluginsToSerialize=./lib/plugin/custom_plugins.dll
```

> [!NOTE]
> 在使用 `--staticPlugins` 和 `--setPluginsToSerialize` 参数构建包含自定义插件的动态模型时，如果遇到错误 `[E] Error[4]: IRuntime::deserializeCudaEngine: Error Code 4: API Usage Error (Cannot register the library as plugin creator of EfficientRotatedNMS_TRT exists already.)`，这通常意味着引擎构建已成功，但在加载并反序列化引擎时检测到插件重复注册。这种情况下，可以忽略该错误。
