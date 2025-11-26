English | [简体中文](README.md)

<div align="center">
  <img width="75%" src="assets/logo.png">

  <p align="center">
      <a href="./LICENSE"><img alt="GitHub License" src="https://img.shields.io/github/license/laugh12321/TensorRT-YOLO?style=for-the-badge&color=0074d9"></a>
      <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img alt="GitHub Release" src="https://img.shields.io/github/v/release/laugh12321/TensorRT-YOLO?style=for-the-badge&color=0074d9"></a>
      <img alt="GitHub Repo Stars" src="https://img.shields.io/github/stars/laugh12321/TensorRT-YOLO?style=for-the-badge&color=3dd3ff">
      <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black">
      <img alt="Arch" src="https://img.shields.io/badge/Arch-x86%20%7C%20ARM-0091BD?style=for-the-badge&logo=cpu&logoColor=white">
      <img alt="NVIDIA" src="https://img.shields.io/badge/NVIDIA-%2376B900.svg?style=for-the-badge&logo=nvidia&logoColor=white">
  </p>
</div>

---

🚀 TensorRT-YOLO is an **easy-to-use**, **extremely efficient** inference deployment tool for the **YOLO series** designed specifically for NVIDIA devices. The project not only integrates TensorRT plugins to enhance post-processing but also utilizes CUDA kernels and CUDA graphs to accelerate inference. TensorRT-YOLO provides support for both C++ and Python inference, aiming to deliver a 📦**out-of-the-box** deployment experience. It covers various task scenarios such as [object detection](examples/detect/), [instance segmentation](examples/segment/), [image classification](examples/classify/), [pose estimation](examples/pose/), [oriented object detection](examples/obb/), and [video analysis](examples/VideoPipe), meeting developers' deployment needs in **multiple scenarios**.

<div align="center">

<img src='assets/task-banner.png' width="800px">
<img src='assets/example.gif' width="800px">

</div>

## <div align="center">🌠 Recent updates</div>

- 2025-10-05: Precision perfectly aligned, CUDA flawlessly replicates LetterBox with a pixel error of 0 in the vast majority of cases. The Python module has undergone significant restructuring, greatly enhancing usability. 🌟 NEW

- 2025-06-09: In C++, only a single header file `trtyolo.hpp` is included, with zero third-party dependencies (no need to link CUDA and TensorRT when using the module). Support for data structures with image spacing (Pitch) has been added. For more details, see [Bilibili](https://www.bilibili.com/video/BV1e2N1zjE3L). 🌟 NEW

- 2025-04-19: Added support for [YOLO-World](https://docs.ultralytics.com/zh/models/yolo-world/) and [YOLOE](https://docs.ultralytics.com/zh/models/yoloe/), including classification, oriented bounding boxes, pose estimation, and instance segmentation. See [Bilibili](https://www.bilibili.com/video/BV12N5bzkENV) for details. 🌟 NEW

- 2025-03-29: Added support for [YOLO12](https://github.com/sunsmarterjie/yolov12), including classification, oriented bounding boxes, pose estimation, and instance segmentation. See [issues](https://github.com/sunsmarterjie/yolov12/issues/22) for details. 🌟 NEW

- [Performance Leap! TensorRT-YOLO 6.0: Comprehensive Upgrade Analysis and Practical Guide](https://medium.com/@laugh12321/performance-leap-tensorrt-yolo-6-0-comprehensive-upgrade-analysis-and-practical-guide-9d19ad3b53f9) 🌟 NEW

## <div align="center">✨ Key Features</div>

### 🎯 Diverse YOLO Support
- **Comprehensive Compatibility**: Supports YOLOv3 to YOLO12 series models, as well as PP-YOLOE, PP-YOLOE+, YOLO-World, and YOLOE, meeting diverse needs. See [🖥️ Supported Models List](#support-models) for details.
- **Flexible Switching**: Provides simple and easy-to-use interfaces for quick switching between different YOLO versions. 🌟 NEW
- **Multi-Scenario Applications**: Offers rich example codes covering [Detect](examples/detect/), [Segment](examples/segment/), [Classify](examples/classify/), [Pose](examples/pose/), [OBB](examples/obb/), and more.

### 🚀 Performance Optimization
- **CUDA Acceleration**: Optimizes pre-processing through CUDA kernels and accelerates inference using CUDA graphs.
- **TensorRT Integration**: Deeply integrates TensorRT plugins to significantly speed up post-processing and improve overall inference efficiency.
- **Multi-Context Inference**: Supports multi-context parallel inference to maximize hardware resource utilization. 🌟 NEW
- **Memory Management Optimization**: Adapts multi-architecture memory optimization strategies (e.g., Zero Copy mode for Jetson) to enhance memory efficiency. 🌟 NEW

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
- **Customizable Preprocessing Parameters**: Supports flexible configuration of various preprocessing parameters, including **channel swapping (SwapRB)**, **normalization parameters**, and **border padding**. 🌟 NEW

## <div align="center">💨 Quick Start</div>

### 1. Prerequisites

- **CUDA**: Recommended version ≥ 11.0.1
- **TensorRT**: Recommended version ≥ 8.6.1
- **Operating System**: Linux (x86_64 or arm) (recommended); Windows is also supported

> [!NOTE]  
> If you are developing on Windows, you can refer to the following setup guides:
>
> - [Windows Development Environment Setup – NVIDIA](https://www.cnblogs.com/laugh12321/p/17830096.html)
> - [Windows Development Environment Setup – C++](https://www.cnblogs.com/laugh12321/p/17827624.html)

### 2. Compilation and Installation

First, clone the TensorRT-YOLO repository:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
```

Then compile using CMake with the following steps:

```bash
pip install "pybind11[global]" # Install pybind11 to generate Python bindings
cmake -S . -B build -D TRT_PATH=/your/tensorrt/dir -D BUILD_PYTHON=ON -D CMAKE_INSTALL_PREFIX=/your/tensorrt-yolo/install/dir
cmake --build build -j$(nproc) --config Release --target install
```

After executing the above commands, the `tensorrt-yolo` library will be installed in the specified `CMAKE_INSTALL_PREFIX` directory. The `include` folder will contain the header files, and the `lib` folder will contain the `trtyolo` dynamic library and the `custom_plugins` dynamic library (only needed when building OBB, Segment, or Pose models with `trtexec`). If the `BUILD_PYTHON` option is enabled during compilation, the corresponding Python binding files will also be generated in the `tensorrt_yolo/libs` path.

> [!NOTE]  
> Before using the C++ dynamic library, ensure that the specified `CMAKE_INSTALL_PREFIX` path is added to the environment variables so that CMake's `find_package` can locate the `tensorrt-yolo-config.cmake` file. This can be done using the following command:
>
> ```bash
> export PATH=$PATH:/your/tensorrt-yolo/install/dir # linux
> $env:PATH = "$env:PATH;C:\your\tensorrt-yolo\install\dir;C:\your\tensorrt-yolo\install\dir\bin" # windows
> ```

If you want to experience the same inference speed in Python as in C++, you need to enable the `BUILD_PYTHON` option during compilation, and then follow the steps below:

```bash
pip install --upgrade build
python -m build --wheel
pip install dist/trtyolo-6.*-py3-none-any.whl
```

### 3. Model Export

- Use the [`trtyolo-export`](https://github.com/laugh12321/TensorRT-YOLO/tree/export) tool package that comes with the project to export the ONNX model suitable for inference in this project and build it into a TensorRT engine.

### 4. Inference Example

- Inference using Python:

  ```python
  import cv2
  import supervision as sv

  from trtyolo import TRTYOLO

  # -------------------- Initialize the model --------------------
  # Note: The task parameter must match the task type specified during export ("detect", "segment", "classify", "pose", "obb")
  # The profile parameter, when enabled, calculates performance metrics during inference, which can be retrieved by calling model.profile()
  # The swap_rb parameter, when enabled, swaps the channel order before inference (ensuring the model input is RGB)
  model = TRTYOLO("yolo11n-with-plugin.engine", task="detect", profile=True, swap_rb=True)

  # -------------------- Load the test image and perform inference --------------------
  image = cv2.imread("test_image.jpg")
  result = model.predict(image)
  print(f"==> result: {result}")

  # -------------------- Visualize the results --------------------
  box_annotator = sv.BoxAnnotator()
  annotated_frame = box_annotator.annotate(scene=image.copy(), detections=result)

  # -------------------- Performance evaluation --------------------
  throughput, cpu_latency, gpu_latency = model.profile()
  print(throughput)
  print(cpu_latency)
  print(gpu_latency)

  # -------------------- Clone the model --------------------
  # Clone the model instance (suitable for multi-threading scenarios)
  cloned_model = model.clone()  # Create an independent copy to avoid resource contention
  # Verify the consistency of inference with the cloned model
  cloned_result = cloned_model.predict(input_img)
  print(f"==> cloned_result: {cloned_result}")
  ```

- Inference using C++:

  ```cpp
  #include <memory>
  #include <opencv2/opencv.hpp>

  #include "trtyolo.hpp"

  int main() {
      try {
          // -------------------- Initialization --------------------
          trtyolo::InferOption option;
          option.enableSwapRB();  // BGR->RGB conversion

          // Special model parameter setup example
          // const std::vector<float> mean{0.485f, 0.456f, 0.406f};
          // const std::vector<float> std{0.229f, 0.224f, 0.225f};
          // option.setNormalizeParams(mean, std);

          // -------------------- Model Initialization --------------------
          // The models ClassifyModel, DetectModel, OBBModel, SegmentModel, and PoseModel correspond to image classification, detection, oriented bounding box, segmentation, and pose estimation models, respectively.
          auto detector = std::make_unique<trtyolo::DetectModel>(
              "yolo11n-with-plugin.engine",  // Model path
              option                         // Inference settings
          );

          // -------------------- Data Loading --------------------
          cv::Mat cv_image = cv::imread("test_image.jpg");
          if (cv_image.empty()) {
              throw std::runtime_error("Failed to load test image.");
          }

          // Encapsulate image data (no pixel data copying)
          trtyolo::Image input_image(
              cv_image.data,     // Pixel data pointer
              cv_image.cols,     // Image width
              cv_image.rows     // Image height
          );

          // -------------------- Inference Execution --------------------
          trtyolo::DetectRes result = detector->predict(input_image);
          std::cout << result << std::endl;

          // -------------------- Result Visualization (Example) --------------------
          // Implement visualization logic in actual development, e.g.:
          // cv::Mat vis_image = visualize_detections(cv_image, result);
          // cv::imwrite("vis_result.jpg", vis_image);

          // -------------------- Model Cloning Demo --------------------
          auto cloned_detector = detector->clone();  // Create an independent instance
          trtyolo::DetectRes cloned_result = cloned_detector->predict(input_image);

          // Verify result consistency
          std::cout << cloned_result << std::endl;

      } catch (const std::exception& e) {
          std::cerr << "Program Exception: " << e.what() << std::endl;
          return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
  }
  ```

### 5. Inference Flowchart

Below is the flowchart of the `predict` method, which illustrates the complete process from input image to output result:

<div>
  <p>
      <img width="100%" src="./assets/flowsheet.png"></a>
  </p>
</div>

Simply pass the image to be inferred to the `predict` method. The `predict` method will automatically complete preprocessing, model inference, and post-processing internally, and output the inference results. These results can be further applied to downstream tasks (such as visualization, object tracking, etc.).

> For more deployment examples, please refer to the [Model Deployment Examples](examples) section.

## <div align="center">🌟 Sponsorship & Support</div>

Open-source projects thrive on support. If this project has been helpful to you, consider sponsoring the author. Your support is the greatest motivation for continued development!

<div align="center">
  <a href="https://afdian.com/a/laugh12321">
    <img width="200" src="https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png" alt="Sponsor Me">
  </a>
</div>

---

🙏 **A Heartfelt Thank You to Our Supporters and Sponsors**:

> [!NOTE]
>
> The following is a list of sponsors automatically generated by GitHub Actions, updated daily ✨.

<div align="center">
  <a target="_blank" href="https://afdian.com/a/laugh12321">
    <img alt="Sponsors List" src="https://github.com/laugh12321/sponsor/blob/main/sponsors.svg?raw=true">
  </a>
</div>

## <div align="center">📄 License</div>

TensorRT-YOLO is licensed under the **GPL-3.0 License**, an [OSI-approved](https://opensource.org/licenses/) open-source license that is ideal for students and enthusiasts, fostering open collaboration and knowledge sharing. Please refer to the [LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) file for more details.

Thank you for choosing TensorRT-YOLO; we encourage open collaboration and knowledge sharing, and we hope you comply with the relevant provisions of the open-source license.

## <div align="center">📞 Contact</div>

For bug reports and feature requests regarding TensorRT-YOLO, please visit [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)!

Giving the project a ⭐ Star helps us prioritize your needs and speed up the response time!

## <div align="center">🙏 Thanks</div>

<div align="center">
<a href="https://hellogithub.com/repository/942570b550824b1b9397e4291da3d17c" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=942570b550824b1b9397e4291da3d17c&claim_uid=2AGzE4dsO8ZUD9R&theme=neutral" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</div>

## <div align="center">🌟 Star History</div>

[![Star History Chart](https://api.star-history.com/svg?repos=laugh12321/TensorRT-YOLO&type=date&legend=top-left)](https://www.star-history.com/#laugh12321/TensorRT-YOLO&type=date&legend=top-left)
