<div align="center">
  <p>
      <img width="100%" src="assets/logo.png"></a>
  </p>

[English](README.en.md) | 简体中文

</div>

## <div align="center">🚀 TensorRT YOLO</div>

<p align="center">
    <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/commits"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/laugh12321/TensorRT-YOLO?style=for-the-badge&color=rgb(47%2C154%2C231)"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2350e472">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2320878f">
</p>

TensorRT-YOLO 是一个支持 YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、PP-YOLOE 和 PP-YOLOE+ 的推理加速项目，使用 NVIDIA TensorRT 进行优化。项目不仅集成了 EfficientNMS TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数以及 CUDA 图来加速推理。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供快速而优化的目标检测解决方案。

<div align="center">
    <img src=assets/example0.jpg height="320">
    <img src=assets/example1.jpg height="320">
</div>

## <div align="center">✨ 主要特性</div>

- 支持 YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、PP-YOLOE 和 PP-YOLOE+
- 支持 ONNX 静态、动态导出以及 TensorRT 推理
- 集成 EfficientNMS TensorRT 插件加速后处理
- 利用 CUDA 核函数加速前处理
- 利用 CUDA 图加速推理流程
- 支持 C++ 和 Python 推理
- CLI 快速导出与推理
- Docker 一键部署

## <div align="center">🛠️ 环境要求</div>

- 推荐 CUDA 版本 >= 11.6
- 推荐 TensorRT 版本 >= 8.6

## <div align="center">📦 使用教程</div>

- [快速编译安装](docs/cn/build_and_install.md)

- [使用 CLI 模型导出](docs/cn/model_export.md)

- [模型推理示例](demo/detect/README.md)

- [视频分析示例](demo/VideoPipe/README.md)

## <div align="center">📺 BiliBili</div>

- [啪的一下，很快啊！TensorRT YOLOv5s 在FP16模式下，批量大小4，仅需13毫秒！](https://www.bilibili.com/video/BV1dy421q7Am)

- [【TensorRT-YOLO】YOLOv9 TensorRT 推理➕EfficientNMS](https://www.bilibili.com/video/BV1uF4m1V7xF)

- [【TensorRT-YOLO】YOLOv8 推理最速传说 1ms](https://www.bilibili.com/video/BV13f421o7KL)

- [【TensorRT-YOLO】3.0 Docker 部署演示](https://www.bilibili.com/video/BV1Jr42137EP)

- [【TensorRT-YOLO】CUDA Graphs 加速推理](https://www.bilibili.com/video/BV1RZ421M7JV)

- [【TensorRT-YOLO】接入 VideoPipe 演示](https://www.bilibili.com/video/BV121421C755)

## <div align="center">📄 许可证</div>

TensorRT-YOLO采用 **GPL-3.0许可证**，这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) 文件以了解更多细节。

感谢您选择使用 TensorRT-YOLO，我们鼓励开放的协作和知识分享，同时也希望您遵守开源许可的相关规定。

## <div align="center">📞 联系方式</div>

对于 TensorRT-YOLO 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)！
