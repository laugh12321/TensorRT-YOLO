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

TensorRT-YOLO 是一个支持 YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、PP-YOLOE 和 PP-YOLOE+ 的推理加速项目，使用 NVIDIA TensorRT 进行优化。项目不仅集成了 TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数以及 CUDA 图来加速推理。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供快速而优化的目标检测解决方案。

<div align="center">
    <img src=assets/example.gif width="643">
</div>

<div align="center">
    <img src=assets/example0.jpg height="320">
    <img src=assets/example1.jpg height="320">
</div>

## <div align="center">✨ 主要特性</div>

- 支持 YOLOv3、YOLOv5、YOLOv6、YOLOv7、YOLOv8、YOLOv9、YOLOv10、YOLO11、PP-YOLOE 和 PP-YOLOE+
- 支持 Detection 与 OBB Detection 模型
- 支持 ONNX 静态、动态导出以及 TensorRT 推理
- 集成 TensorRT 插件加速后处理
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

- [【TensorRT-YOLO】你的YOLO快速部署工具](https://www.bilibili.com/video/BV12T421r7ZH)

- [【TensorRT-YOLO】TensorRT 自定义插件加速 YOLO OBB 部署演示](https://www.bilibili.com/video/BV1NYYze8EST)

- [【TensorRT-YOLO】接入 VideoPipe 演示](https://www.bilibili.com/video/BV121421C755)

- [【TensorRT-YOLO】CUDA Graphs 加速推理](https://www.bilibili.com/video/BV1RZ421M7JV)

- [【TensorRT-YOLO】3.0 Docker 部署演示](https://www.bilibili.com/video/BV1Jr42137EP)

## <div align="center">☕ 请作者喝杯咖啡</div>

开源不易，如果本项目有帮助到你的话，可以考虑请作者喝杯咖啡，你的支持是开发者持续维护的最大动力~

> 推荐使用支付宝，微信获取不到头像。转账请备注【TensorRT-YOLO】。

<div align="center">
    <p>
        <img width="500px" src="assets/sponsor.png"></a>
    </p>
</div>

## <div align="center">📄 许可证</div>

TensorRT-YOLO采用 **GPL-3.0许可证**，这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) 文件以了解更多细节。

感谢您选择使用 TensorRT-YOLO，我们鼓励开放的协作和知识分享，同时也希望您遵守开源许可的相关规定。

## <div align="center">📞 联系方式</div>

对于 TensorRT-YOLO 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)！
