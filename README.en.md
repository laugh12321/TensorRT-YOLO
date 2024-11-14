<div align="center">
  <p>
      <img width="100%" src="assets/logo.png"></a>
  </p>

English | [简体中文](README.md)

</div>

## <div align="center">🚀 TensorRT YOLO</div>

<p align="center">
    <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/commits"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/laugh12321/TensorRT-YOLO?style=for-the-badge&color=rgb(47%2C154%2C231)"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2350e472">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2320878f">
</p>

TensorRT-YOLO is an inference acceleration project that supports YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, PP-YOLOE and PP-YOLOE+ using NVIDIA TensorRT for optimization. The project not only integrates the TensorRT plugin to enhance post-processing effects but also utilizes CUDA kernel functions and CUDA graphs to accelerate inference. TensorRT-YOLO provides support for both C++ and Python inference, aiming to deliver a fast and optimized object detection solution.

<div align="center">
    <img src=assets/example.gif width="643">
</div>

<div align="center">
    <img src=assets/example0.jpg height="320">
    <img src=assets/example1.jpg height="320">
</div>

## <div align="center">✨ Key Features</div>

- Support for YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, PP-YOLOE and PP-YOLOE+
- Support Detect and OBB Detect models
- Support for ONNX static and dynamic export, as well as TensorRT inference
- Integration of TensorRT plugin for accelerated post-processing
- Utilization of CUDA kernel functions for accelerated preprocessing
- Utilization of CUDA graphs for accelerated inference process
- Support for inference in both C++ and Python
- Command-line interface for quick export and inference
- One-click Docker deployment

## <div align="center">🛠️ Requirements</div>

- Recommended CUDA version >= 11.6
- Recommended TensorRT version >= 8.6

## <div align="center">📦 Usage Guide</div>

- [Quick Compile and Install](docs/en/build_and_install.md)

- [Export Models using CLI](docs/en/model_export.md)

- [Model Inference Examples](demo/detect/README.en.md)

- [Video Analysis Example](demo/VideoPipe/README.en.md)

## <div align="center">📺 BiliBili</div>

- [【TensorRT-YOLO】你的YOLO快速部署工具](https://www.bilibili.com/video/BV12T421r7ZH)

- [【TensorRT-YOLO】TensorRT 自定义插件加速 YOLO OBB 部署演示](https://www.bilibili.com/video/BV1NYYze8EST)

- [【TensorRT-YOLO】接入 VideoPipe 演示](https://www.bilibili.com/video/BV121421C755)

- [【TensorRT-YOLO】CUDA Graphs 加速推理](https://www.bilibili.com/video/BV1RZ421M7JV)

- [【TensorRT-YOLO】3.0 Docker 部署演示](https://www.bilibili.com/video/BV1Jr42137EP)

## <div align="center">☕ Buy the Author a Coffee</div>

Open source projects require effort. If this project has been helpful to you, consider buying the author a coffee. Your support is the greatest motivation for the developer to keep maintaining the project!

> It's recommended to use Alipay, as WeChat doesn't provide the avatar. Please note "TensorRT-YOLO" in the transfer.

<div align="center">
    <p>
        <img width="500px" src="assets/sponsor.png"></a>
    </p>
</div>

## <div align="center">📄 License</div>

TensorRT-YOLO is licensed under the **GPL-3.0 License**, an [OSI-approved](https://opensource.org/licenses/) open-source license that is ideal for students and enthusiasts, fostering open collaboration and knowledge sharing. Please refer to the [LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) file for more details.

Thank you for choosing TensorRT-YOLO; we encourage open collaboration and knowledge sharing, and we hope you comply with the relevant provisions of the open-source license.

## <div align="center">📞 Contact</div>

For bug reports and feature requests regarding TensorRT-YOLO, please visit [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)!
