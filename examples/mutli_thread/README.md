[English](README.en.md) | 简体中文

# TensorRT-YOLO 多线程与多进程推理指南

TensorRT-YOLO 为 Python 和 C++ 开发者提供了多线程和多进程推理的示例代码：

- [Python 多线程/多进程推理示例](./mutli_thread_process.py)
- [C++ 多线程推理示例](./mutli_thread.cpp)

## 多线程推理中的模型克隆

TensorRT-YOLO 允许多个线程共享一个 engine 进行推理。一个 engine 可以支持多个 context 同时使用，这意味着多个线程可以共享同一份模型权重和参数，而仅在内存或显存中保留一份副本。因此，尽管复制了多个对象，但模型的内存占用并不会线性增加。

TensorRT-YOLO 提供了以下接口用于模型克隆：

- Python: `DetectModel.clone()`
- C++: `TRTYOLO::clone()`

### Python 示例

```python
import cv2
import supervision as sv

from trtyolo import TRTYOLO

# -------------------- 初始化模型 --------------------
# 注意：task参数需与导出时指定的任务类型一致（"detect"、"segment"、"classify"、"pose"、"obb"）
# profile参数开启后，会在推理时计算性能指标，调用 model.profile() 可获取
# swap_rb参数开启后，会在推理前交换通道顺序（确保模型输入时RGB）
model = TRTYOLO("yolo11n-with-plugin.engine", task="detect", profile=True, swap_rb=True)

# -------------------- 加载测试图片并推理 --------------------
image = cv2.imread("test_image.jpg")
result = model.predict(image)
print(f"==> result: {result}")

# -------------------- 可视化结果 --------------------
box_annotator = sv.BoxAnnotator()
annotated_frame = box_annotator.annotate(scene=image.copy(), detections=result)

# -------------------- 性能评估 --------------------
throughput, cpu_latency, gpu_latency = model.profile()
print(throughput)
print(cpu_latency)
print(gpu_latency)

# -------------------- 克隆模型 --------------------
# 克隆模型实例（适用于多线程场景）
cloned_model = model.clone()  # 创建独立副本，避免资源竞争
# 验证克隆模型推理一致性
cloned_result = cloned_model.predict(input_img)
print(f"==> cloned_result: {cloned_result}")
```

### C++ 示例

```cpp
#include <memory>
#include <opencv2/opencv.hpp>

#include "trtyolo.hpp"

int main() {
    try {
        // -------------------- 初始化配置 --------------------
        trtyolo::InferOption option;
        option.enableSwapRB();  // BGR->RGB转换

        // 特殊模型参数设置示例
        // const std::vector<float> mean{0.485f, 0.456f, 0.406f};
        // const std::vector<float> std{0.229f, 0.224f, 0.225f};
        // option.setNormalizeParams(mean, std);

        // -------------------- 模型初始化 --------------------
        // ClassifyModel、DetectModel、OBBModel、SegmentModel 和 PoseModel 分别对应于图像分类、检测、方向边界框、分割和姿态估计模型
        auto detector = std::make_unique<trtyolo::DetectModel>(
            "yolo11n-with-plugin.engine",  // 模型路径
            option                         // 推理设置
        );

        // -------------------- 数据加载 --------------------
        cv::Mat cv_image = cv::imread("test_image.jpg");
        if (cv_image.empty()) {
            throw std::runtime_error("无法加载测试图片");
        }

        // 封装图像数据（不复制像素数据）
        trtyolo::Image input_image(
            cv_image.data,     // 像素数据指针
            cv_image.cols,     // 图像宽度
            cv_image.rows     // 图像高度
        );

        // -------------------- 执行推理 --------------------
        trtyolo::DetectRes result = detector->predict(input_image);
        std::cout << result << std::endl;

        // -------------------- 结果可视化（示意） --------------------
        // 实际开发需实现可视化逻辑，示例：
        // cv::Mat vis_image = visualize_detections(cv_image, result);
        // cv::imwrite("vis_result.jpg", vis_image);

        // -------------------- 模型克隆演示 --------------------
        auto cloned_detector = detector->clone();  // 创建独立实例
        trtyolo::DetectRes cloned_result = cloned_detector->predict(input_image);

        // 验证结果一致性
        std::cout << cloned_result << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "程序异常: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```

## Python 多线程与多进程推理对比

由于 Python 的全局解释器锁（GIL）限制，在计算密集型任务中，多线程无法充分利用多核 CPU 的性能。因此，Python 提供了多进程和多线程两种并发推理方式。以下是它们的对比：

### 多进程与多线程推理的比较

|     | 资源占用 | 计算密集型任务 | I/O密集型任务 | 进程/线程间通信 |
|:-------|:------|:----------|:----------|:----------|
| 多进程   | 高 | 快 | 快 | 慢|
| 多线程   | 低 | 慢 | 较快 |快|

> [!NOTE]
>
> 以上分析为理论上的对比。实际应用中，由于多进程间的结果汇总涉及进程间通信，且任务类型（计算密集型或I/O密集型）难以明确区分，因此需要根据具体任务进行测试和优化。

## C++ 多线程推理 

C++ 的多线程推理在资源占用和性能方面具有显著优势，是并发推理的最佳选择。

### C++ 模型克隆与不克隆的显存占用对比

**硬件配置**:
- **CPU**: AMD Ryzen 7 5700X 8-Core Processor  
- **GPU**: NVIDIA GeForce RTX 2080 Ti 22GB  
**模型**: YOLO11x

| 模型数 | `model.clone()` | 不克隆模型 |
|:---   |:-----           |:-----    |
|1      |408M             |408M      |
|2      |536M             |716M      |
|3      |662M             |1092M      |
|4      |790M             |1470M      |

> [!IMPORTANT]
>
> 使用 `model.clone()` 后，显存占用仍会有少量增加，主要原因如下：
>
> 1. **输入输出的显存预分配**：
>    - 每个 `context` 需要为模型的输入和输出分配独立的显存缓冲区。即使多个 `context` 共享同一个 `engine`，输入输出缓冲区仍需单独分配，这会占用额外的显存。
>    - 输入输出缓冲区的显存占用取决于模型输入输出的形状和数据类型。对于较大的输入输出，这部分显存占用可能会比较显著。
>
> 2. **Context 的显存开销**：
>    - 每个 `context` 是独立的执行上下文，用于管理模型推理的执行流。即使共享了 `engine`，每个 `context` 仍然需要分配一些额外的显存来存储执行状态、中间结果和临时缓冲区。
>    - `context` 的显存占用通常比 `engine` 小得多，但随着 `context` 数量的增加，显存占用也会逐渐增加。
>
> 3. **CUDA 运行时开销**：
>    - CUDA 运行时需要为每个 `context` 分配一些额外的显存来管理执行流、内核启动和内存操作。
>    - TensorRT 在推理过程中可能会使用一些优化技术（如内存重用、内核融合等），这些优化技术可能会引入额外的显存开销。
