## <div align="center">🚀 TensorRT YOLO</div>

TensorRT-YOLO 是一个支持 YOLOv5、YOLOv8、YOLOv9、PP-YOLOE 和 PP-YOLOE+ 的推理加速项目，使用 NVIDIA TensorRT 进行优化。项目不仅集成了 EfficientNMS TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数来加速前处理过程。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供快速而优化的目标检测解决方案。

<div align="center">
    <img src=assets/example0.jpg height="320">
    <img src=assets/example1.jpg height="320">
</div>

## <div align="center">✨ 主要特性</div>

- 支持 FLOAT32、FLOAT16 ONNX 导出以及TensorRT推理
- 支持 YOLOv5、YOLOv8、YOLOv9、PP-YOLOE 和 PP-YOLOE+
- 集成 EfficientNMS TensorRT 插件加速后处理
- 利用 CUDA 核函数加速前处理
- 支持 C++ 和 Python 推理

## <div align="center">🛠️ 环境要求</div>

- 推荐 CUDA 版本 >= 11.4
- 推荐 TensorRT 版本 >= 8.4

## <div align="center">📦 使用教程</div>

<details open>
<summary>安装</summary>

克隆 repo，并要求在 [**Python>=3.8.0**](https://www.python.org/) 环境中安装 [requirements.txt](https://github.com/laugh12321/TensorRT-YOLO/blob/master/requirements.txt)，且要求 [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/)（导出 YOLOv5、YOLOv8 与 YOLOv9）、[**PaddlePaddle>=2.5**](https://www.paddlepaddle.org.cn/install/quick/)（导出 PP-YOLOE 与 PP-YOLOE+）。

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  # clone
cd TensorRT-YOLO
pip install -r requirements.txt  # install
pip install ultralytics          # Optional, export YOLOv5, YOLOv8 and YOLOv9
pip install paddle2onnx          # Optional, export PP-YOLOE and PP-YOLOE+
```
</details>

<details>
<summary>模型导出</summary>

使用下面的命令将导出 ONNX 模型并添加 [EfficientNMS](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin) 插件进行后处理。

**注意：** 导出 PP-YOLOE 与 PP-YOLOE+ 的 ONNX 模型，输入图片尺寸 `imgsz` 必须与[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)导出的尺寸一致，默认为 `640`。

**YOLOv5**
```bash
python python/export/yolov5/export.py -w yolov5s.pt -o output -b 8 --img 640 -s --half
```

**YOLOv8**
```bash
python python/export/yolov8/export.py -w yolov8s.pt -o output --conf-thres 0.25 --iou-thres 0.45 --max-boxes 100
```

**YOLOv9**
```bash
python python/export/yolov9/export.py -w yolov9-e.pt -o output --conf-thres 0.25 --iou-thres 0.45 --max-boxes 100
```

**PP-YOLOE 与 PP-YOLOE+**
```bash
python python/export/ppyoloe/export.py --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output
```

生成的 ONNX 模型使用 `trtexec` 工具导出 TensorRT 模型。

**注意：** 使用 `python export.py --half` 导出的 ONNX 模型在使用 `trtexec` 时必须加上 `--fp16`。

```bash
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
```
</details>

<details>
<summary>使用 detect.py 推理</summary>

`detect.py` 目前支持对单张图片进行推理或批量推理整个目录，可通过 `--inputs` 参数指定推理数据。推理结果可使用 `--output` 参数指定保存路径，默认为 `None`，表示不保存。有关详细指令描述，请运行`python detect.py -h`查看。

```bash
python detect.py  -e model.engine -o output -i img.jpg                         # image
                                               path/                           # directory
```
</details>

## <div align="center">📄 许可证</div>

TensorRT-YOLO采用 **GPL-3.0许可证**，这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) 文件以了解更多细节。

感谢您选择使用 TensorRT-YOLO，我们鼓励开放的协作和知识分享，同时也希望您遵守开源许可的相关规定。

## <div align="center">📞 联系方式</div>

对于 TensorRT-YOLO 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)！
