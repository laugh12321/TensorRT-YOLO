English | [简体中文](README.md)

<div align="center">
  <p>
      <img width="100%" src="assets/logo.png"></a>
  </p>
</div>

## <div align="center">🚀 TensorRT YOLO</div>

<p align="center">
    <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/commits"><img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/laugh12321/TensorRT-YOLO?style=for-the-badge&color=rgb(47%2C154%2C231)"></a>
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2350e472">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/laugh12321/TensorRT-YOLO?style=for-the-badge&color=%2320878f">
</p>

<p align="center">
    <a href="/docs/cn/build_and_install"><img src="https://img.shields.io/badge/-Installation-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href="/examples/"><img src="https://img.shields.io/badge/-Usage Examples-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href="#quick-start"><img src="https://img.shields.io/badge/-Quick Start-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href=""><img src="https://img.shields.io/badge/-API Documentation-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
    <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img src="https://img.shields.io/badge/-Release Notes-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
</p>

🚀 TensorRT-YOLO is an **easy-to-use**, **extremely efficient** inference deployment tool for the **YOLO series** designed specifically for NVIDIA devices. The project not only integrates TensorRT plugins to enhance post-processing but also utilizes CUDA kernels and CUDA graphs to accelerate inference. TensorRT-YOLO provides support for both C++ and Python inference, aiming to deliver a 📦**out-of-the-box** deployment experience. It covers various task scenarios such as [object detection](examples/detect/), [instance segmentation](examples/segment/), [image classification](examples/classify/), [pose estimation](examples/pose/), [oriented object detection](examples/obb/), and [video analysis](examples/VideoPipe), meeting developers' deployment needs in **multiple scenarios**.

<div align="center">

[<img src='assets/obb.png' height="138px" width="190px">](examples/obb/)
[<img src='assets/detect.jpg' height="138px" width="190px">](examples/detect/)
[<img src='assets/segment.jpg' height="138px" width="190px">](examples/segment/)
[<img src='assets/pose.jpg' height="138px" width="190px">](examples/pose/)
[<img src='assets/example.gif' width="770px">](examples/videopipe)

</div>

## <div align="center">✨ Key Features</div>

### 🎯 Diverse YOLO Support
- **Comprehensive Compatibility**: Supports YOLOv3 to YOLOv11 series models, as well as PP-YOLOE and PP-YOLOE+, meeting diverse needs.
- **Flexible Switching**: Provides simple and easy-to-use interfaces for quick switching between different YOLO versions.
- **Multi-Scenario Applications**: Offers rich example codes covering [Detect](examples/detect/), [Segment](examples/segment/), [Classify](examples/classify/), [Pose](examples/pose/), [OBB](examples/obb/), and more.

### 🚀 Performance Optimization
- **CUDA Acceleration**: Optimizes pre-processing through CUDA kernels and accelerates inference using CUDA graphs.
- **TensorRT Integration**: Deeply integrates TensorRT plugins to significantly speed up post-processing and improve overall inference efficiency.
- **Multi-Context Inference**: Supports multi-context parallel inference to maximize hardware resource utilization.
- **Memory Management Optimization**: Adapts multi-architecture memory optimization strategies (e.g., Zero Copy mode for Jetson) to enhance memory efficiency.

### 🛠️ Usability
- **Out-of-the-Box**: Provides comprehensive C++ and Python inference support to meet different developers' needs.
- **CLI Tools**: Built-in command-line tools for quick model export and inference, improving development efficiency.
- **Docker Support**: Offers one-click Docker deployment solutions to simplify environment configuration and deployment processes.
- **No Third-Party Dependencies**: All functionalities are implemented using standard libraries, eliminating the need for additional dependencies and simplifying deployment.
- **Easy Deployment**: Provides dynamic library compilation support for easy calling and deployment.

### 🌐 Compatibility
- **Multi-Platform Support**: Fully compatible with various operating systems and hardware platforms, including Windows, Linux, ARM, and x86.
- **TensorRT Compatibility**: Perfectly adapts to TensorRT 10.x versions, ensuring seamless integration with the latest technology ecosystem.

### 🔧 Flexible Configuration
- **Customizable Preprocessing Parameters**: Supports flexible configuration of various preprocessing parameters, including **channel swapping (SwapRB)**, **normalization parameters**, and **border padding**.

## <div align="center">🔮 Documentation</div>

- **Installation Guide**
    - [📦 Quick Compilation and Installation](docs/en/build_and_install.md)
- **Usage Examples**
    - [Object Detection Example](examples/detect/README.en.md)
    - [Instance Segmentation Example](examples/segment/README.en.md)
    - [Image Classification Example](examples/classify/README.en.md)
    - [Pose Estimation Example](examples/pose/README.en.md)
    - [Oriented Object Detection Example](examples/obb/README.en.md)
    - [📹 Video Analysis Example](examples/VideoPipe/README.en.md)
- **API Documentation**
    - Python API Documentation (⚠️ Not Implemented)
    - C++ API Documentation (⚠️ Not Implemented)
- **FAQ**
    - ⚠️ Collecting ...
- **Supported Models List**
    - [🖥️ Supported Models List](#support-models)

## <div align="center">💨 Quick Start</div><div id="quick-start"></div>

### 1. Prerequisites

- **CUDA**: Recommended version ≥ 11.0.1
- **TensorRT**: Recommended version ≥ 8.6.1
- **Operating System**: Linux (x86_64 or arm) (recommended); Windows is also supported

### 2. Installation

- Refer to the [📦 Quick Compilation and Installation](docs/en/build_and_install.md) documentation.

### 3. Model Export

- Refer to the [🔧 Model Export](docs/en/model_export.md) documentation to export an ONNX model suitable for inference in this project and build it into a TensorRT engine.

### 4. Inference Example

> [!NOTE]
>
> `ClassifyModel`, `DetectModel`, `OBBModel`, `SegmentModel`, and `PoseModel` correspond to image classification (Classify), detection (Detect), oriented bounding box (OBB), segmentation (Segment), and pose estimation (Pose) models, respectively.

- Inference using Python:

  ```python
  import cv2
  from tensorrt_yolo.infer import InferOption, DetectModel, generate_labels, visualize

  # Configure inference options
  option = InferOption()
  option.enable_swap_rb()

  # Initialize the model
  model = DetectModel("yolo11n-with-plugin.engine", option)

  # Load an image
  im = cv2.imread("test_image.jpg")

  # Model prediction
  result = model.predict(im)
  print(f"==> detect result: {result}")

  # Visualize detection results
  labels = generate_labels("labels.txt")
  vis_im = visualize(im, result, labels)
  cv2.imwrite("vis_image.jpg", vis_im)

  # Clone the model and perform prediction
  clone_model = model.clone()
  clone_result = clone_model.predict(im)
  print(f"==> detect clone result: {clone_result}")
  ```

- Inference using C++:

  ```cpp
  #include <memory>
  #include <opencv2/opencv.hpp>

  // For convenience, the module uses only CUDA and TensorRT, with the rest implemented using standard libraries
  #include "deploy/model.hpp"  // Contains model inference-related class definitions
  #include "deploy/option.hpp"  // Contains inference option configuration class definitions
  #include "deploy/result.hpp"  // Contains inference result definitions

  int main() {
      // Configure inference options
      deploy::InferOption option;
      option.enableSwapRB();  // Enable channel swapping (from BGR to RGB)

      // Initialize the model
      auto model = std::make_unique<deploy::DetectModel>("yolo11n-with-plugin.engine", option);

      // Load an image
      cv::Mat cvim = cv::imread("test_image.jpg");
      deploy::Image im(cvim.data, cvim.cols, cvim.rows);

      // Model prediction
      deploy::DetResult result = model->predict(im);

      // Visualization (code omitted)
      // ...  // Visualization code not provided, can be implemented as needed

      // Clone the model and perform prediction
      auto clone_model = model->clone();
      deploy::DetResult clone_result = clone_model->predict(im);

      return 0;  // Program ends normally
  }
  ```

For more deployment examples, please refer to the [Model Deployment Examples](examples) section.

## <div align="center">🖥️ Model Support List</div><div id="support-models"></div>

<div align="center">
    <table>
        <tr>
            <td>
                <img src='assets/yolo-detect.jpeg' height="300">
                <center>Detect</center>
            </td>
            <td>
                <img src='assets/yolo-segment.jpeg' height="300">
                <center>Segment</center>
            </td>
        </tr>
        <tr>
            <td>
                <img src='assets/yolo-pose.jpeg' height="300">
                <center>Pose</center>
            </td>
            <td>
                <img src='assets/yolo-obb.jpeg' height="300">                                
                <center>OBB</center>
            </td>
        </tr>
    </table>
</div>

Symbol legend: (1)  ✅ : Supported; (2) ❔: In progress; (3) ❎ : Not supported; (4) ❎ : Self-implemented export required for inference. <br>

<div style="text-align: center;">
  <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
      <th style="text-align: center;">Task Scenario</th>
      <th style="text-align: center;">Model</th>
      <th style="text-align: center;">CLI Export</th>
      <th style="text-align: center;">Inference Deployment</th>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/meituan/YOLOv6">meituan/YOLOv6</a></td> 
      <td>❎ Refer to <a href="https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800">official export tutorial</a></td> 
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a></td> 
      <td>❎ Refer to <a href="https://github.com/WongKinYiu/yolov7#export">official export tutorial</a></td> 
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a></td> 
      <td>❎ Refer to <a href="https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461">official export tutorial</a></td> 
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/THU-MIG/yolov10">THU-MIG/yolov10</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/PaddlePaddle/PaddleDetection">PaddleDetection/PP-YOLOE+</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/meituan/YOLOv6/tree/yolov6-seg">meituan/YOLOv6-seg</a></td> 
      <td>❎ Implement yourself referring to <a href="https://github.com/laugh12321/TensorRT-YOLO/blob/main/tensorrt_yolo/export/head.py">tensorrt_yolo/export/head.py</a></td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a></td> 
      <td>❎ Implement yourself referring to <a href="https://github.com/laugh12321/TensorRT-YOLO/blob/main/tensorrt_yolo/export/head.py">tensorrt_yolo/export/head.py</a></td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a></td> 
      <td>❎ Implement yourself referring to <a href="https://github.com/laugh12321/TensorRT-YOLO/blob/main/tensorrt_yolo/export/head.py">tensorrt_yolo/export/head.py</a></td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a></td> 
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Classify</td>
      <td><a href="https://github.com/ultralytics/yolov3">ultralytics/yolov3</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Classify</td>
      <td><a href="https://github.com/ultralytics/yolov5">ultralytics/yolov5</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Classify</td>
      <td><a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Pose</td>
      <td><a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
    <tr>
      <td>OBB</td>
      <td><a href="https://github.com/ultralytics/ultralytics">ultralytics/ultralytics</a></td>
      <td>✅</td>
      <td>✅</td>
    </tr>
  </table>
</div>

## <div align="center">📄 License</div>

TensorRT-YOLO is licensed under the **GPL-3.0 License**, an [OSI-approved](https://opensource.org/licenses/) open-source license that is ideal for students and enthusiasts, fostering open collaboration and knowledge sharing. Please refer to the [LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) file for more details.

Thank you for choosing TensorRT-YOLO; we encourage open collaboration and knowledge sharing, and we hope you comply with the relevant provisions of the open-source license.

## <div align="center">📞 Contact</div>

For bug reports and feature requests regarding TensorRT-YOLO, please visit [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)!

## <div align="center">🙏 Thanks</div>

<center>
<a href="https://hellogithub.com/repository/942570b550824b1b9397e4291da3d17c" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=942570b550824b1b9397e4291da3d17c&claim_uid=2AGzE4dsO8ZUD9R&theme=neutral" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</center>
