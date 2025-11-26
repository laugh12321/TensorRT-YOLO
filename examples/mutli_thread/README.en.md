[简体中文](README.md) | English

# TensorRT-YOLO Multi-threading and Multi-processing Inference Guide

TensorRT-YOLO provides example code for multi-threading and multi-processing inference for both Python and C++ developers:

- [Python Multi-threading/Multi-processing Inference Example](./mutli_thread_process.py)
- [C++ Multi-threading Inference Example](./mutli_thread.cpp)

## Model Cloning in Multi-threading Inference

TensorRT-YOLO allows multiple threads to share a single engine for inference. One engine can support multiple contexts simultaneously, meaning multiple threads can share the same model weights and parameters while maintaining only one copy in memory or GPU memory. As a result, even though multiple objects are cloned, the memory footprint of the model does not increase linearly.

TensorRT-YOLO provides the following interfaces for model cloning:

- Python: `DetectModel.clone()`
- C++: `TRTYOLO::clone()`

### Python Example

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

### C++ Example

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

## Python Multi-threading vs. Multi-processing Inference

Due to Python's Global Interpreter Lock (GIL) limitation, multi-threading cannot fully utilize multi-core CPU performance in compute-intensive tasks. Therefore, Python provides both multi-processing and multi-threading approaches for concurrent inference. Below is a comparison:

### Comparison of Multi-processing and Multi-threading Inference

|     | Resource Usage | Compute-Intensive Tasks | I/O-Intensive Tasks | Inter-process/Thread Communication |
|:-------|:------|:----------|:----------|:----------|
| Multi-processing   | High | Fast | Fast | Slow |
| Multi-threading   | Low | Slow | Relatively Fast | Fast |

> [!NOTE]
>
> The above analysis is theoretical. In practice, since result aggregation across processes involves inter-process communication and task types (compute-intensive or I/O-intensive) are often hard to distinguish, specific tasks require testing and optimization.

## C++ Multi-threading Inference

C++ multi-threading inference has significant advantages in terms of resource usage and performance, making it the best choice for concurrent inference.

### GPU Memory Usage Comparison: Model Cloning vs. No Cloning in C++

**Hardware Configuration**:
- **CPU**: AMD Ryzen 7 5700X 8-Core Processor  
- **GPU**: NVIDIA GeForce RTX 2080 Ti 22GB  
**Model**: YOLO11x

| Number of Models | `model.clone()` | No Cloning |
|:---   |:-----           |:-----    |
|1      |408M             |408M      |
|2      |536M             |716M      |
|3      |662M             |1092M      |
|4      |790M             |1470M      |

> [!IMPORTANT]
>
> After using `model.clone()`, GPU memory usage still increases slightly due to the following reasons:
>
> 1. **Pre-allocation of Input/Output GPU Memory**:
>    - Each `context` requires independent GPU memory buffers for model inputs and outputs. Even though multiple `contexts` share the same `engine`, input/output buffers need to be allocated separately, which consumes additional GPU memory.
>    - The GPU memory usage of input/output buffers depends on the shape and data type of the model's inputs and outputs. For larger inputs and outputs, this memory usage can be significant.
>
> 2. **Context GPU Memory Overhead**:
>    - Each `context` is an independent execution context used to manage the execution flow of model inference. Even though the `engine` is shared, each `context` still requires additional GPU memory to store execution states, intermediate results, and temporary buffers.
>    - The GPU memory usage of a `context` is typically much smaller than that of an `engine`, but it increases gradually as the number of `contexts` grows.
>
> 3. **CUDA Runtime Overhead**:
>    - The CUDA runtime requires additional GPU memory for each `context` to manage execution flows, kernel launches, and memory operations.
>    - TensorRT may use optimization techniques (e.g., memory reuse, kernel fusion) during inference, which can introduce additional GPU memory overhead.
