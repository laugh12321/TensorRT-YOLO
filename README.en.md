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

🚀TensorRT-YOLO is a **user-friendly** and **extremely efficient** inference deployment tool for the **YOLO series**, specifically designed for NVIDIA devices. This project not only integrates TensorRT plugins to enhance post-processing effects but also utilizes CUDA kernels and CUDA graphs to accelerate inference. TensorRT-YOLO provides support for both C++ and Python inference, aiming to offer a **plug-and-play** deployment experience. It includes task scenarios such as [object detection](examples/detect/), [instance segmentation](examples/segment/), [pose recognition](examples/pose/), [oriented bounding box detection](examples/obb/), and [video analysis](examples/VideoPipe), meeting developers' **multi-scenario** deployment needs.

<div align="center">
    <img src="assets/example.gif" width="800">
</div>

<div align="center">
    <table>
        <tr>
            <td>
                <img src='assets/detect.jpg' height="300">
                <center>Detect</center>
            </td>
            <td>
                <img src='assets/segment.jpg' height="300">
                <center>Segment</center>
            </td>
        </tr>
        <tr>
            <td>
                <img src='assets/pose.jpg' height="300">
                <center>Pose</center>
            </td>
            <td>
                <img src='assets/obb.png' height="300">                                
                <center>OBB</center>
            </td>
        </tr>
    </table>
</div>

## <div align="center">✨ Key Features</div>

- **Diverse YOLO Support**: Fully compatible with YOLOv3 to YOLOv11, as well as PP-YOLOE and PP-YOLOE+, meeting the needs of different versions.
- **Multi-scenario Applications**: Provides example code for diverse scenarios such as [Detect](examples/detect/), [Segment](examples/segment/), [Pose](examples/pose/), and [OBB](examples/obb/).
- **Model Optimization and Inference Acceleration**:
  - **ONNX Support**: Supports static and dynamic export of ONNX models, including TensorRT custom plugin support, simplifying the model deployment process.
  - **TensorRT Integration**: Integrated TensorRT plugins, including custom plugins, accelerate post-processing for Detect, Segment, Pose, OBB, and other scenarios, enhancing inference efficiency.
  - **CUDA Acceleration**: Optimizes pre-processing with CUDA kernels and accelerates inference processes with CUDA graph technology, achieving high-performance computing.
- **Language Support**: Supports C++ and Python (mapped through Pybind11, enhancing Python inference speed), meeting the needs of different programming languages.
- **Deployment Convenience**:
  - **Dynamic Library Compilation**: Provides support for dynamic library compilation, facilitating calling and deployment.
  - **No Third-Party Dependencies**: All features are implemented using standard libraries, with no additional dependencies, simplifying the deployment process.
- **Rapid Development and Deployment**:
  - **CLI Tools**: Provides a command-line interface (CLI) tool for quick model export and inference.
  - **Cross-Platform Support**: Supports various devices such as Windows, Linux, ARM, x86, adapting to different hardware environments.
  - **Docker Deployment**: Supports one-click deployment with Docker, simplifying environment configuration and deployment processes.
- **TensorRT Compatibility**: Compatible with TensorRT 10.x versions, ensuring compatibility with the latest technologies.

## <div align="center">🔮 Documentation and Tutorials</div>

- **Installation Guide**
    - [📦 Quick Compilation and Installation](docs/en/build_and_install.md)
- **Quick Start**
    - [✴️ Python SDK Quick Start](#quick-start-python)  
    - [✴️ C++ SDK Quick Start](#quick-start-cpp)
- **Usage Examples**
    - [Object Detection Example](examples/detect/README.md)
    - [Instance Segmentation Example](examples/segment/README.md)
    - [Pose Recognition Example](examples/pose/README.md)
    - [Oriented Bounding Box Detection Example](examples/obb/README.md)
    - [📹 Video Analysis Example](examples/VideoPipe/README.md)
- **API Documentation**
    - Python API Documentation (⚠️ Not Implemented)
    - C++ API Documentation (⚠️ Not Implemented)
- **Frequently Asked Questions**
    - ⚠️ Collecting...
- **Model Support List**
    - [🖥️ Model Support List](#support-models)

## <div align="center">💨 Quick Start</div>

### 🔸 Prerequisites

- Recommended CUDA version >= 11.0.1 (Minimum CUDA version 11.0.1)
- Recommended TensorRT version >= 8.6.1 (Minimum TensorRT version 8.6.1)
- OS: Linux x86_64 (Recommended) arm / Windows /

### 🎆 Quick Installation

- Refer to the [📦 Quick Compilation and Installation](docs/en/build_and_install.md) documentation

### Python SDK Quick Start<div id="quick-start-python"></div>

> [!IMPORTANT]
> Before inference, please refer to the [🔧 CLI Model Export](/docs/en/model_export.md) documentation to export the ONNX model suitable for this project's inference and build it into a TensorRT engine.

#### Python CLI Inference Example

> [!NOTE] 
> Using the `--cudaGraph` option can significantly improve inference speed, but note that this feature is only available for static models.
> 
> The `-m, --mode` parameter can be used to select different model types, where `0` represents detection (Detect), `1` represents oriented bounding box (OBB), `2` represents segmentation (Segment), and `3` represents pose estimation (Pose).

1. Use the `tensorrt_yolo` library's `trtyolo` command-line tool for inference. Run the following command to view help information:

    ```bash
    trtyolo infer --help
    ```

2. Run the following command for inference:

    ```bash
    trtyolo infer -e models/yolo11n.engine -m 0 -i images -o output -l labels.txt --cudaGraph
    ```

    Inference results will be saved to the `output` folder, and visualized results will be generated.

#### Python Inference Example

> [!NOTE] 
> `DeployDet`, `DeployOBB`, `DeploySeg`, and `DeployPose` correspond to detection (Detect), oriented bounding box (OBB), segmentation (Segment), and pose estimation (Pose) models, respectively.
>
> For these models, the `CG` version utilizes CUDA Graph to further accelerate the inference process, but please note that this feature is limited to static models.

```python
import cv2
from tensorrt_yolo.infer import DeployCGDet, DeployDet, generate_labels_with_colors, visualize

use_cudaGraph = True
engine_path = "yolo11n-with-plugin.engine"
model = DeployCGDet(engine_path) if use_cudaGraph else DeployDet(engine_path)

im = cv2.imread("test_image.jpg")
result = model.predict(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) # The model accepts images in RGB format
print(f"==> detect result: {result}")

# Visualization
labels = generate_labels_with_colors("labels.txt")
vis_im = visualize(image, result, labels)
cv2.imwrite("vis_image.jpg", vis_im)

```

### C++ SDK Quick Start<div id="quick-start-cpp"></div>

> [!IMPORTANT]
> Before inference, please refer to the [🔧 CLI Model Export](/docs/en/model_export.md) documentation to export the ONNX model suitable for this project's inference and build it into a TensorRT engine.

> [!NOTE] 
> `DeployDet`, `DeployOBB`, `DeploySeg`, and `DeployPose` correspond to detection (Detect), oriented bounding box (OBB), segmentation (Segment), and pose estimation (Pose) models, respectively.
>
> For these models, the `CG` version utilizes CUDA Graph to further accelerate the inference process, but please note that this feature is limited to static models.


```cpp
#include <opencv2/opencv.hpp>
// For ease of use, the module uses standard libraries in addition to CUDA and TensorRT
#include "deploy/vision/inference.hpp"
#include "deploy/vision/result.hpp"

int main(int argc, char* argv[]) {
    bool useCudaGraph = true;
    deploy::DeployBase model;
    if (useCudaGraph) {
        model = deploy::DeployCGDet("yolo11n-with-plugin.engine");
    } else {
        model = deploy::DeployDet("yolo11n-with-plugin.engine");
    }
    auto cvim = cv::imread("test_image.jpg");

    cv::cvtColor(cvim, cvim, cv::COLOR_BGR2RGB);
    deploy::Image im(cvim.data, cvim.cols, cvim.rows); // The model accepts images in RGB format
    deploy::DetResult result = model.predict(im);

    // Visualization
    // ...

    return 0;
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
      <td>OBB</td>
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
  </table>
</div>

## <div align="center">☕ Buy the Author a Coffee</div>

Open source projects require effort. If this project has been helpful to you, consider buying the author a coffee. Your support is the greatest motivation for the developer to keep maintaining the project!

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
