[简体中文](README.md) | English

# TensorRT-YOLO Multi-threading and Multi-processing Inference Guide

TensorRT-YOLO provides example code for multi-threading and multi-processing inference for both Python and C++ developers:

- [Python Multi-threading/Multi-processing Inference Example](./mutli_thread_process.py)
- [C++ Multi-threading Inference Example](./mutli_thread.cpp)

## Model Cloning in Multi-threading Inference

TensorRT-YOLO allows multiple threads to share a single engine for inference. One engine can support multiple contexts simultaneously, meaning multiple threads can share the same model weights and parameters while maintaining only one copy in memory or GPU memory. As a result, even though multiple objects are cloned, the memory footprint of the model does not increase linearly.

TensorRT-YOLO provides the following interfaces for model cloning (using DetectModel as an example):

- Python: `DetectModel.clone()`
- C++: `DetectModel::clone()`

### Python Example

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

# Perform model prediction
result = model.predict(im)
print(f"==> Detection result: {result}")

# Visualize the detection results
labels = generate_labels("labels.txt")
vis_im = visualize(im, result, labels)
cv2.imwrite("vis_image.jpg", vis_im)

# Clone the model and perform prediction
clone_model = model.clone()
clone_result = clone_model.predict(im)
print(f"==> Cloned model detection result: {clone_result}")
```

### C++ Example

```cpp
#include <memory>
#include <opencv2/opencv.hpp>

#include "trtyolo.hpp"

int main() {
    // Configure inference options
    trtyolo::InferOption option;
    option.enableSwapRB();  // Enable channel swapping (BGR to RGB)

    // Initialize the model
    auto model = std::make_unique<trtyolo::DetectModel>("yolo11n-with-plugin.engine", option);

    // Load an image
    cv::Mat cvim = cv::imread("test_image.jpg");
    trtyolo::Image im(cvim.data, cvim.cols, cvim.rows);

    // Perform model prediction
    trtyolo::DetResult result = model->predict(im);

    // Visualization (code omitted)
    // ...  // Visualization code is not provided and can be implemented as needed

    // Clone the model and perform prediction
    auto clone_model = model->clone();
    trtyolo::DetResult clone_result = clone_model->predict(im);

    return 0;  // Program ends successfully
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
