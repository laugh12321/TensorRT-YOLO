[English](README.en.md) | 简体中文

## <div align="center">🚀 TensorRT YOLO</div>

<p align="center">
    <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/commits"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/laugh12321/TensorRT-YOLO?style=for-the-badge&color=rgb(47%2C154%2C231)"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2350e472">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2320878f">
</p>

TensorRT-YOLO 是一个支持 YOLOv5、YOLOv8、YOLOv9、PP-YOLOE 和 PP-YOLOE+ 的推理加速项目，使用 NVIDIA TensorRT 进行优化。项目不仅集成了 EfficientNMS TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数来加速前处理过程。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供快速而优化的目标检测解决方案。

<div align="center">
    <img src=assets/example0.jpg height="320">
    <img src=assets/example1.jpg height="320">
</div>

## <div align="center">✨ 主要特性</div>

- 支持 YOLOv5、YOLOv8、YOLOv9、PP-YOLOE 和 PP-YOLOE+
- 支持 ONNX 静态、动态导出以及 TensorRT 推理
- 集成 EfficientNMS TensorRT 插件加速后处理
- 利用 CUDA 核函数加速前处理 (V1.0)
- 支持 C++ 和 Python 推理（C++ 实现中）

## <div align="center">🛠️ 环境要求</div>

- 推荐 CUDA 版本 >= 11.7
- 推荐 TensorRT 版本 >= 8.6

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

**注意：** 导出 PP-YOLOE 与 PP-YOLOE+ 的 ONNX 模型，只会对 `batch` 维度进行修改，`height` 与 `width` 维度无法被更改，需要在[PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)中设置，默认为 `640`。

**YOLOv5, v8, v9**

```bash
# Static
python python/export/{yolo version}/export.py -w your_model_path.pt -o output -b 8 --img 640 -s
# Dynamic
python python/export/{yolo version}/export.py -w your_model_path.pt -o output -s --dynamic
```

**PP-YOLOE 与 PP-YOLOE+**

```bash
# Static
python python/export/ppyoloe/export.py --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output -b 8 -s
# Dynamic
python python/export/ppyoloe/export.py --model_dir modeldir --model_filename model.pdmodel --params_filename model.pdiparams -o output -s --dynamic
```

生成的 ONNX 模型使用 `trtexec` 工具导出 TensorRT 模型。

```bash
# Static
trtexec --onnx=model.onnx --saveEngine=model.engine --fp16
# Dynamic
trtexec --onnx=model.onnx --saveEngine=model.engine --minShapes=images:1x3x640x640 --optShapes=images:4x3x640x640 --maxShapes=images:8x3x640x640 --fp16
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
