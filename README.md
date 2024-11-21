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

🚀TensorRT-YOLO 是一款专为 NVIDIA 设备设计的**易用灵活**、**极致高效**的**YOLO系列**推理部署工具。项目不仅集成了 TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数以及 CUDA 图来加速推理。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供📦**开箱即用**的部署体验。包括 [目标检测](examples/detect/)、[实例分割](examples/segment/)、[姿态识别](examples/pose/)、[旋转目标检测](examples/obb/)、[视频分析](examples/VideoPipe)等任务场景，满足开发者**多场景**的部署需求。


<div align="center">
    <img src=assets/example.gif width="800">
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

## <div align="center">✨ 主要特性</div>

- **多样化的YOLO支持**：全面兼容YOLOv3至YOLOv11以及PP-YOLOE和PP-YOLOE+，满足不同版本需求。
- **多场景应用**：提供[Detect](examples/detect/)、[Segment](examples/segment/)、[Pose](examples/pose/)、[OBB](examples/obb/)等多样化场景的示例代码。
- **模型优化与推理加速**：
  - **ONNX支持**：支持ONNX模型的静态和动态导出，包括TensorRT自定义插件支持，简化模型部署流程。
  - **TensorRT集成**：集成TensorRT插件，包括自定义插件，加速Detect, Segment, Pose, OBB等场景的后处理，提升推理效率。
  - **CUDA加速**：利用CUDA核函数优化前处理，CUDA图技术加速推理流程，实现高性能计算。
- **语言支持**：支持C++和Python（通过Pybind11映射，提升Python推理速度），满足不同编程语言需求。
- **部署便捷性**：
  - **动态库编译**：提供动态库编译支持，方便调用和部署。
  - **无第三方依赖**：全部功能使用标准库实现，无需额外依赖，简化部署流程。
- **快速开发与部署**：
  - **CLI工具**：提供命令行界面（CLI）工具，实现快速模型导出和推理。
  - **跨平台支持**：支持Windows、Linux、ARM、x86等多种设备，适应不同硬件环境。
  - **Docker部署**：支持Docker一键部署，简化环境配置和部署流程。
- **TensorRT兼容性**：兼容TensorRT 10.x版本，确保与最新技术兼容。

## <div align="center">🔮 文档教程</div>


- **安装文档**
    - [📦 快速编译安装](docs/cn/build_and_install.md)
- **快速开始**
    - [✴️ Python SDK快速使用](#quick-start-python)  
    - [✴️ C++ SDK快速使用](#quick-start-cpp)
- **使用示例**
    - [目标检测 示例](examples/detect/README.md)
    - [实例分割 示例](examples/segment/README.md)
    - [姿态识别 示例](examples/pose/README.md)
    - [旋转目标检测 示例](examples/obb/README.md)
    - [📹视频分析 示例](examples/VideoPipe/README.md)
- **API文档**
    - Python API文档（⚠️ 未实现）
    - C++ API文档（⚠️ 未实现）
- **常见问题**
    - ⚠️ 收集中 ...
- **模型支持列表**
    - [🖥️ 模型支持列表](#support-models)

## <div align="center">💨 快速开始</div>

### 🔸 前置依赖

- 推荐 CUDA 版本 >= 11.6
- 推荐 TensorRT 版本 >= 8.6.1 （TensorRT 最低版本 8.6.1）
- OS: Linux x86_64 (推荐) arm / Windows /

### 🎆 快速安装

- 参考[📦 快速编译安装](docs/cn/build_and_install.md)文档

### Python SDK快速开始<div id="quick-start-python"></div>

#### Python CLI 推理示例

> [!NOTE] 
> 使用 `--cudaGraph` 选项可以显著提升推理速度，但需知此功能仅适用于静态模型。
> 
> 通过 `-m, --mode` 参数可以选择不同的模型类型，其中 `0` 代表检测（Detect）、`1` 代表旋转边界框（OBB）、`2` 代表分割（Segment）、`3` 代表姿态估计（Pose）。

1. 使用 `tensorrt_yolo` 库的 `trtyolo` 命令行工具进行推理。运行以下命令查看帮助信息：

    ```bash
    trtyolo infer --help
    ```

2. 运行以下命令进行推理：

    ```bash
    trtyolo infer -e models/yolo11n.engine -m 0 -i images -o output -l labels.txt --cudaGraph
    ```

    推理结果将保存至 `output` 文件夹，并生成可视化结果。

#### Python 推理示例

> [!NOTE] 
> `DeployDet`、`DeployOBB`、`DeploySeg` 和 `DeployPose` 分别对应于检测（Detect）、方向边界框（OBB）、分割（Segment）和姿态估计（Pose）模型。
>
> 对于这些模型，`CG` 版本利用 CUDA Graph 来进一步加速推理过程，但请注意，这一功能仅限于静态模型。

```python
import cv2
from tensorrt_yolo.infer import DeployCGDet, DeployDet, generate_labels_with_colors, visualize

use_cudaGraph = True
engine_path = "yolo11n-with-plugin.engine"
model = DeployCGDet(engine_path) if use_cudaGraph else DeployDet(engine_path)

im = cv2.imread("test_image.jpg")
result = model.predict(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) # model 接收的图片必须是RGB格式
print(f"==> detect result: {result}")

# 可视化
labels = generate_labels_with_colors("labels.txt")
vis_im = visualize(image, result, labels)
cv2.imwrite("vis_image.jpg", vis_im)

```

### C++ SDK快速开始<div id="quick-start-cpp"></div>

> [!IMPORTANT]
> 在进行推理之前，请参考[🔧 CLI 导出模型](/docs/cn/model_export.md)文档，导出适用于该项目推理的ONNX模型并构建为TensorRT引擎。

> [!NOTE] 
> `DeployDet`、`DeployOBB`、`DeploySeg` 和 `DeployPose` 分别对应于检测（Detect）、方向边界框（OBB）、分割（Segment）和姿态估计（Pose）模型。
>
> 对于这些模型，`CG` 版本利用 CUDA Graph 来进一步加速推理过程，但请注意，这一功能仅限于静态模型。


```cpp
#include <opencv2/opencv.hpp>
// 为了方便调用，模块除使用 CUDA、TensorRT 其余均使用标准库实现
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
    deploy::Image im(cvim.data, cvim.cols, cvim.rows); // model 接收的图片必须是RGB格式
    deploy::DetResult result = model.predict(im);

    // 可视化
    // ...

    return 0;
}
```

更多部署案例请参考[模型部署示例](examples) .

## <div align="center">🖥️ 模型支持列表</div><div id="support-models"></div>

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

符号说明: (1)  ✅ : 已经支持; (2) ❔: 正在进行中; (3) ❎ : 暂不支持; (4) 🟢 : 导出自行实现，即可推理. <br>

<div style="text-align: center;">
  <table border="1" style="border-collapse: collapse; width: 100%;">
    <tr>
      <th style="text-align: center;">任务场景</th>
      <th style="text-align: center;">模型</th>
      <th style="text-align: center;">CLI 导出</th>
      <th style="text-align: center;">推理部署</th>
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
      <td>❎ 参考<a href="https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800">官方导出教程</a></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a></td>
      <td>❎ 参考<a href="https://github.com/WongKinYiu/yolov7#export">官方导出教程</a></td>
      <td>✅</td>
    </tr>
    <tr>
      <td>Detect</td>
      <td><a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a></td>
      <td>❎ 参考<a href="https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461">官方导出教程</a></td>
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
      <td>❎ 参考<a href="https://github.com/laugh12321/TensorRT-YOLO/blob/main/tensorrt_yolo/export/head.py">tensorrt_yolo/export/head.py</a> 自行实现</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/WongKinYiu/yolov7">WongKinYiu/yolov7</a></td>
      <td>❎ 参考<a href="https://github.com/laugh12321/TensorRT-YOLO/blob/main/tensorrt_yolo/export/head.py">tensorrt_yolo/export/head.py</a> 自行实现</td>
      <td>🟢</td>
    </tr>
    <tr>
      <td>Segment</td>
      <td><a href="https://github.com/WongKinYiu/yolov9">WongKinYiu/yolov9</a></td>
      <td>❎ 参考<a href="https://github.com/laugh12321/TensorRT-YOLO/blob/main/tensorrt_yolo/export/head.py">tensorrt_yolo/export/head.py</a> 自行实现</td>
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

## <div align="center">☕ 请作者喝杯咖啡</div>

开源不易，如果本项目有帮助到你的话，可以考虑请作者喝杯咖啡，你的支持是开发者持续维护的最大动力~

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
