[English](README.en.md) | 简体中文

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

  <p align="center">
      <a href="/docs/cn/build_and_install.md"><img src="https://img.shields.io/badge/-安装-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="/examples/"><img src="https://img.shields.io/badge/-使用示例-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="#quick-start"><img src="https://img.shields.io/badge/-快速开始-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href=""><img src="https://img.shields.io/badge/-API文档-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
      <a href="https://github.com/laugh12321/TensorRT-YOLO/releases"><img src="https://img.shields.io/badge/-更新日志-0078D4?style=for-the-badge&logo=github&logoColor=white"></a>
  </p>
</div>

---

🚀 TensorRT-YOLO 是一款专为 NVIDIA 设备设计的**易用灵活**、**极致高效**的**YOLO系列**推理部署工具。项目不仅集成了 TensorRT 插件以增强后处理效果，还使用了 CUDA 核函数以及 CUDA 图来加速推理。TensorRT-YOLO 提供了 C++ 和 Python 推理的支持，旨在提供📦**开箱即用**的部署体验。包括 [目标检测](examples/detect/)、[实例分割](examples/segment/)、[图像分类](examples/classify/)、[姿态识别](examples/pose/)、[旋转目标检测](examples/obb/)、[视频分析](examples/VideoPipe)等任务场景，满足开发者**多场景**的部署需求。

<div align="center">

[<img src='assets/obb.png' height="138px" width="190px">](examples/obb/)
[<img src='assets/detect.jpg' height="138px" width="190px">](examples/detect/)
[<img src='assets/segment.jpg' height="138px" width="190px">](examples/segment/)
[<img src='assets/pose.jpg' height="138px" width="190px">](examples/pose/)
[<img src='assets/example.gif' width="770px">](examples/videopipe)

</div>

## <div align="center">🌠 近期更新</div>

- [性能飞跃！TensorRT-YOLO 6.0 全面升级解析与实战指南](https://www.cnblogs.com/laugh12321/p/18693017) 🌟 NEW


## <div align="center">✨ 主要特性</div>

### 🎯 多样化的 YOLO 支持
- **全面兼容**：支持 YOLOv3 至 YOLOv11 全系列模型，以及 PP-YOLOE 和 PP-YOLOE+，满足多样化需求。
- **灵活切换**：提供简洁易用的接口，支持不同版本 YOLO 模型的快速切换。🌟 NEW
- **多场景应用**：提供丰富的示例代码，涵盖[Detect](examples/detect/)、[Segment](examples/segment/)、[Classify](examples/classify/)、[Pose](examples/pose/)、[OBB](examples/obb/)等多种应用场景。

### 🚀 性能优化
- **CUDA 加速**：通过 CUDA 核函数优化前处理流程，并采用 CUDA 图技术加速推理过程。
- **TensorRT 集成**：深度集成 TensorRT 插件，显著加速后处理，提升整体推理效率。
- **多 Context 推理**：支持多 Context 并行推理，最大化硬件资源利用率。🌟 NEW
- **显存管理优化**：适配多架构显存优化策略（如 Jetson 的 Zero Copy 模式），提升显存效率。🌟 NEW

### 🛠️ 易用性
- **开箱即用**：提供全面的 C++ 和 Python 推理支持，满足不同开发者需求。
- **CLI 工具**：内置命令行工具，支持快速模型导出与推理，提升开发效率。
- **Docker 支持**：提供 Docker 一键部署方案，简化环境配置与部署流程。
- **无第三方依赖**：全部功能使用标准库实现，无需额外依赖，简化部署流程。
- **部署便捷**：提供动态库编译支持，方便调用和部署。

### 🌐 兼容性
- **多平台支持**：全面兼容 Windows、Linux、ARM、x86 等多种操作系统与硬件平台。
- **TensorRT 兼容**：完美适配 TensorRT 10.x 版本，确保与最新技术生态无缝衔接。

### 🔧 灵活配置
- **预处理参数自定义**：支持多种预处理参数灵活配置，包括 **通道交换 (SwapRB)**、**归一化参数**、**边界值填充**。🌟 NEW

## <div align="center">🚀 性能对比</div>

<div align="center">

| Model | Official + trtexec (ms) | trtyolo + trtexec (ms) | TensorRT-YOLO Inference (ms)|
|:-----:|:-----------------------:|:----------------------:|:---------------------------:|
| YOLOv11n | 1.611 ± 0.061        | 1.428 ± 0.097          | 1.228 ± 0.048               |
| YOLOv11s | 2.055 ± 0.147        | 1.886 ± 0.145          | 1.687 ± 0.047               |
| YOLOv11m | 3.028 ± 0.167        | 2.865 ± 0.235          | 2.691 ± 0.085               |
| YOLOv11l | 3.856 ± 0.287        | 3.682 ± 0.309          | 3.571 ± 0.102               |
| YOLOv11x | 6.377 ± 0.487        | 6.195 ± 0.482          | 6.207 ± 0.231               |

</div>

> [!NOTE]
>
> **测试环境**
> - **GPU**：NVIDIA RTX 2080 Ti 22GB
> - **输入尺寸**：640×640 像素
>
> **测试工具**
> - **Official**：使用 Ultralytics 官方导出的 ONNX 模型。
> - **trtyolo**：使用 TensorRT-YOLO 提供的 CLI 工具 (trtyolo) 导出的带 EfficientNMS 插件的 ONNX 格式模型。
> - **trtexec**：使用 NVIDIA 的 `trtexec` 工具将 ONNX 模型构建为引擎并进行推理测试。
>   - **构建指令**：`trtexec --onnx=xxx.onnx --saveEngine=xxx.engine --fp16`
>   - **测试指令**：`trtexec --avgRuns=1000 --useSpinWait --loadEngine=xxx.engine`
> - **TensorRT-YOLO Inference**：使用 TensorRT-YOLO 框架对 **trtyolo + trtexec** 方式得到的引擎进行推理的延迟时间（包括前处理、推理和后处理）。

## <div align="center">🔮 文档教程</div>

- **安装指南**
    - [📦 快速编译安装](docs/cn/build_and_install.md)
- **使用示例**
    - [目标检测 示例](examples/detect/README.md)
    - [实例分割 示例](examples/segment/README.md)
    - [图像分类 示例](examples/classify/README.md)
    - [姿态识别 示例](examples/pose/README.md)
    - [旋转目标检测 示例](examples/obb/README.md)
    - [📹视频分析 示例](examples/VideoPipe/README.md)
    - [多线程多进程 示例](examples/mutli_thread/README.md) 🌟 NEW
- **API文档**
    - Python API文档（⚠️ 未实现）
    - C++ API文档（⚠️ 未实现）
- **常见问题**
    - ⚠️ 收集中 ...
- **模型支持列表**
    - [🖥️ 模型支持列表](#support-models)

## <div align="center">💨 快速开始</div><div id="quick-start"></div>

### 1. 前置依赖

- **CUDA**：推荐版本 ≥ 11.0.1
- **TensorRT**：推荐版本 ≥ 8.6.1
- **操作系统**：Linux (x86_64 或 arm)（推荐）；Windows 亦可支持

### 2. 安装

- 参考 [📦 快速编译安装](docs/cn/build_and_install.md) 文档。

### 3. 模型导出

- 参考 [🔧 模型导出](docs/cn/model_export.md) 文档，导出适用于该项目推理的ONNX模型并构建为TensorRT引擎。

### 4. 推理示例

> [!NOTE]
>
> `ClassifyModel`、`DetectModel`、`OBBModel`、`SegmentModel` 和 `PoseModel` 分别对应于图像分类（Classify）、检测（Detect）、方向边界框（OBB）、分割（Segment）、姿态估计（Pose）和模型。

- 使用 Python 进行推理：

  ```python
  import cv2
  from tensorrt_yolo.infer import InferOption, DetectModel, generate_labels, visualize

  def main():
      # -------------------- 初始化配置 --------------------
      # 配置推理设置
      option = InferOption()
      option.enable_swap_rb()  # 将OpenCV默认的BGR格式转为RGB格式
      # 特殊模型配置示例（如PP-YOLOE系列需取消下方注释）
      # option.set_normalize_params([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

      # -------------------- 模型初始化 --------------------
      # 加载TensorRT引擎文件（注意检查文件路径）
      # 提示：首次加载引擎可能需要较长时间进行优化
      model = DetectModel(engine_path="yolo11n-with-plugin.engine", 
                        option=option)

      # -------------------- 数据预处理 --------------------
      # 加载测试图片（建议添加文件存在性检查）
      input_img = cv2.imread("test_image.jpg")
      if input_img is None:
          raise FileNotFoundError("测试图片加载失败，请检查文件路径")

      # -------------------- 执行推理 --------------------
      # 执行目标检测（返回结果包含边界框、置信度、类别信息）
      detection_result = model.predict(input_img)
      print(f"==> detection_result: {detection_result}")

      # -------------------- 结果可视化 --------------------
      # 加载类别标签（需确保labels.txt与模型匹配）
      class_labels = generate_labels(labels_file="labels.txt")
      # 生成可视化结果
      visualized_img = visualize(
          image=input_img,
          result=detection_result,
          labels=class_labels,
      )
      cv2.imwrite("vis_image.jpg", visualized_img)

      # -------------------- 模型克隆演示 --------------------
      # 克隆模型实例（适用于多线程场景）
      cloned_model = model.clone()  # 创建独立副本，避免资源竞争
      # 验证克隆模型推理一致性
      cloned_result = cloned_model.predict(input_img)
      print(f"==> cloned_result: {cloned_result}")

  if __name__ == "__main__":
      main()
  ```

- 使用 C++ 进行推理：

  ```cpp
  #include <memory>
  #include <opencv2/opencv.hpp>

  // 为了方便调用，模块除使用CUDA、TensorRT外，其余均使用标准库实现
  #include "deploy/model.hpp"  // 包含模型推理相关的类定义
  #include "deploy/option.hpp"  // 包含推理选项的配置类定义
  #include "deploy/result.hpp"  // 包含推理结果的定义

  int main() {
      try {
          // -------------------- 初始化配置 --------------------
          deploy::InferOption option;
          option.enableSwapRB();  // BGR->RGB转换

          // 特殊模型参数设置示例
          // const std::vector<float> mean{0.485f, 0.456f, 0.406f};
          // const std::vector<float> std{0.229f, 0.224f, 0.225f};
          // option.setNormalizeParams(mean, std);

          // -------------------- 模型初始化 --------------------
          auto detector = std::make_unique<deploy::DetectModel>(
              "yolo11n-with-plugin.engine",  // 模型路径
              option                         // 推理设置
          );

          // -------------------- 数据加载 --------------------
          cv::Mat cv_image = cv::imread("test_image.jpg");
          if (cv_image.empty()) {
              throw std::runtime_error("无法加载测试图片");
          }

          // 封装图像数据（不复制像素数据）
          deploy::Image input_image(
              cv_image.data,     // 像素数据指针
              cv_image.cols,     // 图像宽度
              cv_image.rows,     // 图像高度
          );

          // -------------------- 执行推理 --------------------
          deploy::DetResult result = detector->predict(input_image);
          std::cout << result << std::endl;

          // -------------------- 结果可视化（示意） --------------------
          // 实际开发需实现可视化逻辑，示例：
          // cv::Mat vis_image = visualize_detections(cv_image, result);
          // cv::imwrite("vis_result.jpg", vis_image);

          // -------------------- 模型克隆演示 --------------------
          auto cloned_detector = detector->clone();  // 创建独立实例
          deploy::DetResult cloned_result = cloned_detector->predict(input_image);

          // 验证结果一致性
          std::cout << cloned_resul << std::endl;

      } catch (const std::exception& e) {
          std::cerr << "程序异常: " << e.what() << std::endl;
          return EXIT_FAILURE;
      }
      return EXIT_SUCCESS;
  }
  ```

### 5.推理流程图

以下是`predict`方法的流程图，展示了从输入图片到输出结果的完整流程：

<div>
  <p>
      <img width="100%" src="./assets/flowsheet.png"></a>
  </p>
</div>

只需将待推理的图片传递给 `predict` 方法，`predict` 内部会自动完成预处理、模型推理和后处理，并输出推理结果，这些结果可进一步应用于下游任务（如可视化、目标跟踪等）。


> 更多部署案例请参考[模型部署示例](examples) .

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

## <div align="center">🌟 赞助与支持</div>

开源不易，如果本项目对你有所帮助，欢迎通过赞助支持作者。你的支持是开发者持续维护的最大动力！

<div align="center">
  <a href="https://afdian.com/a/laugh12321">
    <img width="200" src="https://pic1.afdiancdn.com/static/img/welcome/button-sponsorme.png" alt="赞助我">
  </a>
</div>

---

🙏 **衷心感谢以下支持者与赞助商的无私支持**：

> [!NOTE]
>
> 以下是 GitHub Actions 自动生成的赞助者列表，每日更新 ✨。

<div align="center">
  <a target="_blank" href="https://afdian.com/a/laugh12321">
    <img alt="赞助者列表" src="https://github.com/laugh12321/sponsor/blob/main/sponsors.svg?raw=true">
  </a>
</div>

## <div align="center">📄 许可证</div>

TensorRT-YOLO采用 **GPL-3.0许可证**，这个[OSI 批准](https://opensource.org/licenses/)的开源许可证非常适合学生和爱好者，可以推动开放的协作和知识分享。请查看[LICENSE](https://github.com/laugh12321/TensorRT-YOLO/blob/master/LICENSE) 文件以了解更多细节。

感谢您选择使用 TensorRT-YOLO，我们鼓励开放的协作和知识分享，同时也希望您遵守开源许可的相关规定。

## <div align="center">📞 联系方式</div>

对于 TensorRT-YOLO 的错误报告和功能请求，请访问 [GitHub Issues](https://github.com/laugh12321/TensorRT-YOLO/issues)！

## <div align="center">🙏 致谢</div>

<div align="center">
<a href="https://hellogithub.com/repository/942570b550824b1b9397e4291da3d17c" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=942570b550824b1b9397e4291da3d17c&claim_uid=2AGzE4dsO8ZUD9R&theme=neutral" alt="Featured｜HelloGitHub" style="width: 250px; height: 54px;" width="250" height="54" /></a>
</div>
