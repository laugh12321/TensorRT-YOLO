# TensorRT-YOLO 代码阅读指南

> 适用于版本 6.4.0，帮助你快速理解项目结构和各模块职责。

---

## 一、项目概览

TensorRT-YOLO 是专为 NVIDIA 设备设计的高性能 YOLO 系列推理部署工具。通过 CUDA 核函数加速前处理（Letterbox）、CUDA Graph 加速推理、TensorRT 插件加速后处理（NMS），同时提供 C++ 和 Python 两种接口。

**支持的任务类型：**

| 任务 | 目录 | 对应模型类 |
|------|------|-----------|
| 目标检测 Detect | `examples/detect/` | `DetectModel` |
| 实例分割 Segment | `examples/segment/` | `SegmentModel` |
| 图像分类 Classify | `examples/classify/` | `ClassifyModel` |
| 姿态估计 Pose | `examples/pose/` | `PoseModel` |
| 旋转目标检测 OBB | `examples/obb/` | `OBBModel` |

---

## 二、项目目录结构

```
TensorRT-YOLO/
├── CMakeLists.txt              # 顶层 CMake 构建脚本
├── pyproject.toml              # Python 包配置
├── modules/                    # 【核心 C++ 代码】
│   ├── trtyolo/                # 主推理库
│   │   ├── core/               # TensorRT 引擎管理 + CUDA 图 + 内存管理
│   │   │   ├── core.hpp        # TRTManager / CudaGraph 声明
│   │   │   ├── core.cpp        # TRTManager / CudaGraph 实现
│   │   │   ├── buffer.hpp      # 4种内存策略（Device/Discrete/Unified/Mapped）
│   │   │   └── buffer.cpp      # Buffer 实现
│   │   ├── infer/              # 推理后端 + 模型类 + Letterbox 前处理
│   │   │   ├── trtyolo.hpp     # 数据结构 + 5种模型类 + 配置类声明
│   │   │   ├── trtyolo.cpp     # 5种模型 + 后处理实现
│   │   │   ├── backend.hpp     # TrtBackend 推理后端声明
│   │   │   ├── backend.cpp     # 推理流程：static/dynamic infer + CUDA Graph
│   │   │   ├── letterbox.hpp   # Letterbox CUDA 核函数声明
│   │   │   └── letterbox.cu    # Letterbox GPU 实现
│   │   ├── utils/              # 工具函数
│   │   │   ├── common.hpp      # 配置结构体 / 计时器 / 性能统计
│   │   │   └── common.cpp      # 工具函数实现
│   │   └── binding/            # Python 绑定（pybind11）
│   │       └── trtyolo.cpp     # C++ → Python 导出
│   └── plugin/                 # 【TensorRT 自定义插件】
│       ├── efficientIdxNMSPlugin/        # 高效 NMS 插件（Detect/Pose/Segment）
│       │   ├── efficientIdxNMSPlugin.h   # 插件类声明
│       │   ├── efficientIdxNMSPlugin.cpp # 插件注册 + enqueue
│       │   ├── efficientIdxNMSParameters.h # 参数结构体
│       │   └── efficientIdxNMSInference.cu # GPU NMS 实现
│       ├── efficientRotatedNMSPlugin/    # 旋转框 NMS 插件（OBB）
│       │   ├── efficientRotatedNMSPlugin.h
│       │   ├── efficientRotatedNMSPlugin.cpp
│       │   ├── efficientRotatedNMSParameters.h
│       │   └── efficientRotatedNMSInference.cu
│       └── common/                       # 插件公共代码
├── trtyolo/                    # 【Python 包】
│   ├── __init__.py             # TRTYOLO 用户接口
│   └── c_lib_wrap.py.in        # C++ pybind 模块的 import 模板
├── examples/                   # 【示例代码】
│   ├── detect/                 # 目标检测（C++ / Python）
│   ├── segment/                # 实例分割
│   ├── classify/               # 图像分类
│   ├── pose/                   # 姿态估计
│   ├── obb/                    # 旋转目标检测
│   ├── mutli_thread/           # 多线程推理
│   ├── VideoPipe/              # 视频分析管道
│   └── nndeploy/               # nndeploy 框架集成
└── assets/                     # 图片资源
```

---

## 三、建议阅读顺序

推荐按以下顺序阅读代码，从数据结构到完整流程逐步深入：

### 第一层：数据结构与接口（先建立全局认知）

#### ① `modules/trtyolo/infer/trtyolo.hpp`

这是项目的 **"目录文件"**，先看这里了解有哪些东西：

**图像输入：**
```cpp
struct Image {
    void*  ptr;        // 图像数据指针
    int    width, height, channels;
    size_t pitch;      // 行距（字节），支持 padding
};
```

**基础结果（所有结果类型的基类）：**
```cpp
struct BaseRes {
    int                num;      // 检测数量
    std::vector<int>   classes;  // 类别
    std::vector<float> scores;   // 置信度
};
```

**5 种结果类型（继承 BaseRes）：**

| 结构体 | 额外字段 | 用途 |
|--------|----------|------|
| `ClassifyRes` | 无 | 分类 top-k |
| `DetectRes` | `vector<Box> boxes` | 目标检测 |
| `OBBRes` | `vector<RotatedBox> boxes` | 旋转框检测 |
| `SegmentRes` | `vector<Box> boxes` + `vector<Mask> masks` | 实例分割 |
| `PoseRes` | `vector<Box> boxes` + `vector<vector<KeyPoint>> kpts` | 姿态估计 |

**辅助结构体：**
- `Box` — 矩形框 `{left, top, right, bottom}`，提供 `xyxy()` 方法
- `RotatedBox : Box` — 多一个 `theta` 角度，提供 `xyxyxyxy()` 转8点
- `Mask` — 分割掩码 `{data, width, height}`
- `KeyPoint` — 关键点 `{x, y, conf}`

**配置类 InferOption：**
```cpp
class InferOption {
    void setDeviceId(int);           // GPU 设备 ID
    void enableSwapRB();             // BGR ↔ RGB 交换
    void enablePerformanceReport();  // 开启性能统计
    void setBorderValue(float);      // Letterbox 填充值
    void setNormalizeParams(mean, std);  // 自定义归一化参数
    void setInputDimensions(w, h);   // 固定输入尺寸
    void enableCudaMem();            // 数据已在 GPU 显存
    void enableManagedMemory();      // 统一内存
};
```

**5 种模型类（继承 BaseModel）：**

| 类 | predict 返回 |
|----|-------------|
| `ClassifyModel` | `ClassifyRes` |
| `DetectModel` | `DetectRes` |
| `OBBModel` | `OBBRes` |
| `SegmentModel` | `SegmentRes` |
| `PoseModel` | `PoseRes` |

每个模型类都支持：
- `predict(const Image&)` — 单图推理
- `predict(const vector<Image>&)` — 批量推理
- `clone()` — 克隆实例（共享 engine，独立 context）
- `batch()` — 获取最大 batch size
- `performanceReport()` — 获取吞吐量 / CPU延迟 / GPU延迟

#### ② `modules/trtyolo/utils/common.hpp`

工具层，定义了随处理中传递的配置和计时器：

```cpp
// 预处理参数（传给 CUDA Letterbox kernel）
struct ProcessConfig {
    bool   swap_rb;       // 是否交换 R/B 通道
    float  border_value;  // 填充值（默认 114）
    float3 alpha;         // 归一化乘数（默认 1/255）
    float3 beta;          // 归一化偏移（默认 0）
};

// 推理总配置
struct InferConfig {
    int                 device_id;
    bool                cuda_mem;                  // 输入是否已在 GPU
    bool                enable_managed_memory;     // 统一内存
    bool                enable_performance_report;
    std::optional<int2> input_shape;               // 固定输入尺寸（可选）
    ProcessConfig       config;                    // 预处理参数
};
```

---

### 第二层：推理后处理（理解输出如何变成结果）

#### ③ `modules/trtyolo/infer/trtyolo.cpp`

这里包含每个模型类的 **`predict()`** 和 **后处理** 实现。核心模式：

```
predict(images) → withPerformanceReport() → backend_->infer(images) → postProcess*(idx)
```

**5 个后处理函数的共通过程**（以 Detect 为例）：

```cpp
DetectRes postProcessDetect(int idx) {
    // 1. 从 5 个输出张量读取数据（按 idx 偏移）
    int    num     = num_buffer[idx];           // 该图检测到 num 个框
    float* boxes   = box_buffer + idx * ...;    // 框坐标 [n, 4]
    float* scores  = score_buffer + idx * ...;  // 分数 [n]
    int*   classes = class_buffer + idx * ...;  // 类别 [n]

    // 2. 用 Transform 变换将预测坐标映射回原图像素坐标
    transform.apply(left, top, &left, &top);
    transform.apply(right, bottom, &right, &bottom);

    // 3. 填充结果
    result.boxes.emplace_back(Box{left, top, right, bottom});
}
```

每种后处理的区别：
- **Detect**: 处理 `boxes[4]`
- **OBB**: 处理 `boxes[5]`（多 theta），用 RotatedBox
- **Segment**: 处理 `boxes[4]` + `masks[n, H, W]`，memcpy mask 数据
- **Pose**: 处理 `boxes[4]` + `kpts[n, m, c]`，对每个关键点做坐标变换
- **Classify**: 直接读 top-k `[score, class_id]` 对

**Transform 的反变换** 是后处理的关键：Letterbox 前处理时缩放+填充了图片，`Transform::apply()` 将模型输出的归一化坐标反算回原始图像坐标。

---

### 第三层：推理流程（理解 GPU 上发生了什么）

#### ④ `modules/trtyolo/infer/backend.cpp`

**TrtBackend** 是整个推理的核心调度器。

**构造函数流程：**
```
TrtBackend(engine_file, infer_config):
    1. cudaSetDevice(device_id)
    2. cudaStreamCreate(&stream)
    3. 判断是否支持 Zero Copy (Jetson)
    4. 创建 TRTManager → 读取 engine 文件 → initialize()
    5. getTensorInfo()          ← 获取所有 I/O 张量的 name/shape/dtype
    6. initialize()             ← 分配内存、初始化 Transform
    7. 如果是静态模型 → captureCudaGraph()
```

**静态推理 vs 动态推理：**

| | 静态模型（dynamic=false） | 动态模型（dynamic=true） |
|---|---|---|
| 输入形状 | 固定（如 640×640） | 可变（batch/shape 可调） |
| 执行路径 | `staticInfer()` | `dynamicInfer()` |
| 加速方式 | CUDA Graph（一次捕获，反复 launch） | 传统逐步执行 |
| 节点更新 | `updateKernelNodeParams()` | 不需要 |
| 适用场景 | 监控视频等固定分辨率 | 通用场景 |

**`captureCudaGraph()` 做了什么：**
```
1. 预执行一次（设置地址 + enqueueV3 + sync）
2. beginCapture(stream)           ← 开始记录 CUDA 操作
3. H2D（如果不是 cuda_mem）       ← 主机→设备拷贝
4. cudaLetterbox()                ← GPU 前处理
5. enqueueV3()                    ← TensorRT 推理（含插件 NMS）
6. D2H（所有输出张量）            ← 设备→主机拷贝
7. endCapture(stream)             ← 结束记录，生成 CUDA Graph
8. initializeNodes()              ← 获取图中各节点引用
```

后续每次推理时只需：
1. `updateKernelNodeParams()` — 更新输入图像指针/参数
2. `updateMemcpyNodeParams()` — 更新拷贝参数
3. `cuda_graph_.launch(stream)` — 一键执行全部

**`dynamicInfer()` 流程：**
```
1. 更新所有张量的 shape.d[0] = batch_size
2. manager_->setTensorAddress() + setInputShape()
3. H2D 拷贝输入数据到 GPU
4. 逐个执行 cudaLetterbox()
5. manager_->enqueueV3(stream)     ← TensorRT 推理
6. D2H 拷贝输出到主机
7. cudaStreamSynchronize()
```

#### ⑤ `modules/trtyolo/infer/letterbox.cu`

**`cudaLetterbox()` GPU 核函数**，每个线程处理一个像素：

1. **计算源图坐标** → 根据 `Transform` 的 scale 和偏移，反算输出像素对应源图哪个位置
2. **双线性插值** → 取 4 个邻居像素做加权平均
3. **SwapRB**（可选）→ 通道交换
4. **归一化** → `pixel * alpha + beta`
5. **边界填充** → 超出源图范围时填 `border_value`

精度说明：使用 CUDA 的 `__half2float` 等内置函数，与 Python OpenCV 的 letterbox **像素误差为 0**。

**Transform 结构体：**
```cpp
struct Transform {
    int4  meta;    // (有效宽, 有效高, offset_x, offset_y)
    float scale;   // 缩放比例
    void update(src_w, src_h, dst_w, dst_h);  // 根据尺寸变化更新
    void apply(x, y, &out_x, &out_y);         // 后处理时坐标反变换
};
```

---

### 第四层：引擎管理与 GPU 基础设施

#### ⑥ `modules/trtyolo/core/core.hpp` + `core.cpp`

**TRTManager** — TensorRT 引擎的 RAII 封装：

```cpp
class TRTManager {
    unique_ptr<IExecutionContext> context_;  // 推理上下文
    shared_ptr<ICudaEngine>       engine_;   // CUDA 引擎（shared 支持 clone 共享）
    unique_ptr<IRuntime>          runtime_;  // TensorRT 运行时
    unique_ptr<TRTLogger>         logger_;   // 日志

    void initialize(blob, size);             // 加载 engine 二进制
    unique_ptr<TRTManager> clone();          // 克隆（复刻 engine，创建新 context）
    bool setTensorAddress(name, data);       // 绑定张量内存
    bool setInputShape(name, dims);          // 设置动态输入形状
    bool enqueueV3(stream);                  // 执行推理
};
```

**CudaGraph** — CUDA Graph 生命周期管理：

```cpp
class CudaGraph {
    void beginCapture(stream);               // 开始捕获
    void endCapture(stream);                 // 结束捕获并实例化
    void launch(stream);                     // 执行图
    void initializeNodes(num);               // 获取图中各节点
    void updateKernelNodeParams(idx, params); // 更新 kernel 节点参数
    void updateMemcpyNodeParams(idx, ...);    // 更新 memcpy 节点参数
    void destroy();                           // 释放资源
};
```

#### ⑦ `modules/trtyolo/core/buffer.hpp` + `buffer.cpp`

**4 种内存策略**，通过 `BufferFactory` 自动选择：

| 类型 | 适用场景 | 内存位置 |
|------|----------|----------|
| `DeviceBuffer` | 纯 GPU 内存（输入/输出张量） | 仅 device |
| `DiscreteBuffer` | 标准主机+设备分离 | host + device |
| `UnifiedBuffer` | `enableManagedMemory` 开启时 | 统一寻址 |
| `MappedBuffer` | Jetson 集成显卡 Zero Copy | 映射内存 |

选择逻辑（`getTensorInfo()` 中）：
```
if enable_managed_memory → Unified
else if zero_copy (Jetson) → Mapped
else → Discrete
```

---

### 第五层：插件与 Python 绑定

#### ⑧ `modules/plugin/`

NMS 后处理以 **TensorRT Plugin** 的形式嵌入 engine 中，推理时在 GPU 上直接完成 NMS。

每个插件包含：
- `*Plugin.h/cpp` — 实现 `IPluginV2DynamicExt` 接口（注册、序列化、enqueue）
- `*Parameters.h` — 参数（iou_threshold, score_threshold 等）
- `*Inference.cu/cuh` — CUDA 核函数（实际计算）

efficientIdxNMS 做了什么：
1. 对 raw boxes 按 score 排序
2. 迭代选择最高分框，计算 IoU
3. 抑制重叠框（IoU > threshold）
4. 输出过滤后的 boxes / scores / classes

#### ⑨ `modules/trtyolo/binding/trtyolo.cpp`

pybind11 绑定，将 C++ 类导出到 Python：

- `bind_result<T>()` — 结果类 → Python 类，`.xyxy` `.confidence` `.class_id` `.masks` `.kpts` 等 numpy 属性
- `bind_model<T>()` — 模型类 → Python 类，`.predict()` `.clone()` `.profile()` 方法
- `PyArray2Image()` — `np.ndarray (HWC uint8)` → `trtyolo::Image`

#### ⑩ `trtyolo/__init__.py`

**TRTYOLO** — Python 用户的唯一入口：

```python
model = TRTYOLO(engine_path, task="detect", swap_rb=True, profile=True)
result = model.predict(image)     # sv.Detections
throughput, cpu_lat, gpu_lat = model.profile()
model2 = model.clone()            # 多线程安全
```

`predict()` 内部流程：
1. 支持 `str/Path/ndarray/list` 多种输入
2. 自动按 `batch` 分组
3. 调用 C++ `model.predict()`（走上面所有优化路径）
4. `convert_to_sv()` 转为 supervision 格式

---

## 四、核心概念详解

### 4.1 YOLO 网络结构：Backbone / Neck / Head / NMS

TensorRT-YOLO 推理时，`.engine` 文件内封装了完整的 YOLO 网络。这四部分是深度学习模型本身的架构概念，不是项目代码概念。

#### Backbone（主干网络）— "看清楚图片里有什么"

逐层提取图像特征，从低级特征（边缘、纹理）到高级特征（物体部件、语义）。

```
输入: [640, 640, 3]
     │
┌────▼────┐  第1层：检测边缘、颜色块
│ Conv+BN │
│ +SiLU   │
└────┬────┘
┌────▼────┐  第2-4层：检测纹理、局部形状
│ C2f块×3 │
└────┬────┘
┌────▼────┐  第5-7层：检测物体部件（轮子、窗户）
│ C2f块×6 │
└────┬────┘
┌────▼────┐  第8-10层：检测完整物体轮廓
│ C2f块×6 │
└────┬────┘
┌────▼────┐  第11层：多尺度池化，扩大感受野
│  SPPF   │
└────┬────┘
     │
输出: P3(80×80), P4(40×40), P5(20×20) 三层特征图
```

不同 YOLO 版本的 backbone：
- YOLOv5/v8/v11 → CSPDarknet
- YOLOv9 → GELAN
- YOLOv10 → 改进版 CSPNet

#### Neck（颈部网络）— "把不同尺度信息融合起来"

Backbone 输出的特征图分辨率不同（80×80 细节多但语义弱，20×20 语义强但细节少）。Neck 负责**双向融合**，让每一层都既有细节又有语义。

标准的 FPN + PAN 结构：

```
   P5 ─────────────→ C4 ─────────────→ C3     ← 自顶向下（FPN）：语义传下去
                     │                  │
                     ▼                  ▼
   P5' ←─────────── P4' ←─────────── P3'      ← 自底向上（PAN）：细节传上来
```

#### Head（检测头）— "根据特征给出判断"

在 Neck 输出的特征图上每个 grid cell 预测目标：

```
每个 grid cell 预测：
  • bbox 分支  → [x, y, w, h] 框的位置和大小
  • cls 分支   → [p1, p2, ..., p80] 每个类别的概率
  • seg 分支   → mask 原型系数（仅分割模型）
  • kpt 分支   → 关键点坐标（仅姿态模型）

三种尺度分工：
  P3'(80×80) → 小物体（远处的行人、交通标志）
  P4'(40×40) → 中等物体（车、行人）
  P5'(20×20) → 大物体（公交车、建筑）

总计: 80² + 40² + 20² = 8400 个候选框
```

#### NMS（非极大值抑制）— "去重，只留最好的"

Head 输出的 8400 个候选框，同一物体可能被多个框重复检测。NMS 算法筛选：

```
1. 按置信度排序所有框
2. 取最高分框，加入"最终结果"
3. 计算其余框与它的 IoU（重叠比例）
4. 移除 IoU > 0.5 的框（认为是同一物体）
5. 重复 2-4，直到所有框被处理或丢弃
```

```
同一辆车被检测 3 次：
  ┌──────────────────┐
  │  ┌────────────┐  │
  │  │  ┌──────┐  │  │
  │  │  │ Car  │  │  │
  │  │  │0.92  │  │  │  ← 最高分，保留
  │  │  └──────┘  │  │
  │  │  Car 0.85  │  │  ← IoU 高，抑制
  │  └────────────┘  │
  │    Car 0.78      │  ← IoU 高，抑制
  └──────────────────┘
```

> **在本项目中的体现**：backbone + neck + head 是 TensorRT 将 ONNX 编译成的 CUDA kernel（黑盒），NMS 是 `modules/plugin/` 中自己写的 CUDA 插件嵌入 engine。

---

### 4.2 TrtBackend（推理后端）— 推理流水线的"调度中心"

`TrtBackend`（`backend.hpp/cpp`）不自己做计算，而是负责**把整个推理流水线串起来**。

```
TrtBackend 的职责：
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  ① 持有 engine + context (TRTManager)            │
  │  ② 持有 CUDA stream                              │
  │  ③ 管理输入/输出 buffer (显存分配)                 │
  │  ④ 管理 Transform (前处理参数)                    │
  │  ⑤ 调用 cudaLetterbox          ← 前处理          │
  │  ⑥ 调用 manager_->enqueueV3()  ← 推理 ★          │
  │  ⑦ 调用 deviceToHost           ← 结果搬回 CPU    │
  │                                                  │
  │  ⑧ 如果是静态模型，把 ⑤⑥⑦ 捕获成 CUDA Graph     │
  │    提供 staticInfer / dynamicInfer 两条执行路径    │
  │                                                  │
  └──────────────────────────────────────────────────┘
```

**为什么叫 Backend？** 因为它是底层实现，和前端接口解耦：

```
前端（你调用的）           后端（干活的）
───────────────         ───────────────
DetectModel             →  TrtBackend
  .predict(image)              .infer(images)
                               ├─ cudaLetterbox()     预处理
                               ├─ enqueueV3()         推理
                               └─ deviceToHost()      搬运结果
```

打个比方：DetectModel 是餐厅服务员（接单、上菜），TrtBackend 是后厨（洗菜、炒菜、装盘），TensorRT engine 是灶台（真正加热）。

---

### 4.3 CUDA Graph — GPU 指令的"录制 + 播放"

正常调用 CUDA kernel 时，每次都要经过 CPU → GPU 的提交开销：

```
每次推理（传统方式）：
  CPU: "GPU，跑 letterbox"    → 提交开销 ~10µs
  CPU: "GPU，跑 enqueueV3"    → 提交开销 ~10µs
  CPU: "GPU，结果拷回来"      → 提交开销 ~10µs
  总计：3 次提交，~30µs 开销
```

CUDA Graph 的思想：**录制一次，反复播放**。把所有 kernel 调用序列打包成一张"指令磁带"。

```
初始化（只做一次）：
  beginCapture → letterbox → enqueueV3 → memcpy → endCapture
  Graph = 录制好的"指令磁带"

每次推理：
  Graph.launch()  → 一次性提交全部，~3µs 开销
```

> **和 TaskFlow 的区别**：TaskFlow 在 CPU 上编排线程任务，CUDA Graph 在 GPU 上编排 kernel 指令。思想相似，层次不同。

在本项目中，`CudaGraph` 类（`core.hpp/cpp`）管理整个生命周期：

```cpp
// ===== 录制（captureCudaGraph，初始化时一次） =====
cuda_graph_.beginCapture(stream);
  cudaLetterbox(...);       // 前处理
  manager_->enqueueV3();    // 推理
  deviceToHost(...);        // 结果拷贝
cuda_graph_.endCapture(stream);

// ===== 播放（staticInfer，每次推理） =====
cuda_graph_.updateKernelNodeParams(0, 图片参数);  // 更新参数
cuda_graph_.launch(stream);  // 一键执行
```

静态模型 vs 动态模型的选择：

| | 静态模型（CUDA Graph） | 动态模型 |
|---|---|---|
| 输入形状 | 固定（如 640×640） | 可变 |
| 加速方式 | 录制→播放 | 逐步执行 |
| 适用场景 | 监控视频等固定分辨率 | 通用场景 |

---

### 4.4 updateKernelNodeParams — CUDA Graph 的"换参数"机制

CUDA Graph 一旦捕获，图中所有内存地址就"冻住"了。但每次推理的图片不同、地址不同。`updateKernelNodeParams` 负责在 launch 之前**更新图中 kernel 节点的参数指针**，无需重新捕获整张图。

```cpp
// staticInfer() 中，每次推理更新 Letterbox kernel 参数
for (int idx = 0; idx < num; ++idx) {
    void* kernelParams[] = {
        (void*)&inputs[idx].ptr,      // ← 新图片的 GPU 指针（变）
        (void*)&inputs[idx].width,    // ← 新图片的宽（变）
        (void*)&inputs[idx].height,   // ← 新图片的高（变）
        (void*)&inputs[idx].pitch,    // ← 新图片的行距（变）
        (void*)&infer_device_ptr,     // ← 输出位置（不变）
        (void*)&max_shape,            // ← 模型输入尺寸（不变）
        (void*)&transforms.meta,      // ← 变换参数（不变）
        (void*)&infer_config.config   // ← 预处理配置（不变）
    };
    cuda_graph_.updateKernelNodeParams(idx, kernelParams);
}
```

```
初始化时（capture）：
  Node[0]: cudaLetterbox(图片A地址, ...)  ← 参数被冻结为"图片A"

每次推理时（staticInfer）：
  updateKernelNodeParams(0, 图片B地址)   ← 换成"图片B"
  launch()                                ← 直接执行，无需重捕获
```

**为什么这样设计？** 捕获 CUDA Graph 很贵（几十毫秒），更新参数几乎零开销。这就是静态模型比动态模型快的核心原因。

打个比方：CUDA Graph 像已经填好地址的信封，每次寄信只需把信纸（图片指针）换掉，不需要重新写信封。`updateKernelNodeParams` 就是"换信纸"的动作。

---

### 4.5 cudaLetterbox — GPU 并行的图像预处理

YOLO 模型要求固定尺寸输入（如 640×640），但实际图片尺寸各异。`cudaLetterbox` 在 GPU 上完成等比缩放 + 填充 + 归一化。

**为什么放 GPU？** 避免数据搬运：

```
CPU 方式（慢）：
  图片在 GPU → 拷回 CPU → OpenCV resize → 拷回 GPU → 推理
              ↑ 两次搬运 ~2ms 是瓶颈 ↑

CUDA 方式（快）：
  图片在 GPU → cudaLetterbox → 直接推理
              ↑ 零搬运，GPU 自己搞定 ↑
```

**每个像素并行做 4 件事：**

```
     任意尺寸图片              固定尺寸 (640×640)
  ┌─────────────────┐        ┌──────────────────┐
  │                 │        │  ░░░░░░░░░░░░░░  │ ← 灰色：填充 (border_value)
  │   一辆车         │ ───→  │  ░░░░░░░░░░░░░░  │
  │                 │        │  ░░┌────────┐░░  │
  │  1920×1080      │        │  ░░│  车    │░░  │ ← 中间：等比缩放
  └─────────────────┘        │  ░░└────────┘░░  │
                             │  ░░░░░░░░░░░░░░  │
                             └──────────────────┘

每个像素的处理步骤：
  ① 等比缩放  → 双线性插值，从源图 4 个邻居像素加权平均
  ② 通道交换  → BGR ↔ RGB（可选，SwapRB）
  ③ 归一化    → pixel * alpha + beta（默认 /255，可自定义 mean/std）
  ④ 边界填充  → 缩略图外区域填 border_value（默认 114）
```

精度：与 Python OpenCV 的 letterbox **像素误差为 0**。

---

## 五、整体推理流程图

```
                    用户调用 predict(image)
                           │
                           ▼
              ┌─────────────────────────┐
              │     前处理 (GPU)         │
              │  cudaLetterbox()        │
              │  ┌───────────────────┐  │
              │  │ 等比缩放 + 填充    │  │
              │  │ 归一化 (α*p + β)  │  │
              │  │ SwapRB (可选)     │  │
              │  │ 计算 Transform    │  │
              │  └───────────────────┘  │
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │   TensorRT 推理          │
              │   enqueueV3()           │
              │   ┌───────────────────┐ │
              │   │ YOLO backbone    │ │
              │   │ YOLO neck        │ │
              │   │ YOLO head        │ │
              │   │ NMS Plugin (GPU) │ │  ← 插件已嵌入 engine
              │   └───────────────────┘ │
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │     后处理 (CPU)         │
              │  postProcess*()         │
              │  ┌───────────────────┐  │
              │  │ 读取输出张量       │  │
              │  │ Transform.apply() │  │  ← 坐标反变换
              │  │ 填充结果结构体    │  │
              │  └───────────────────┘  │
              └───────────┬─────────────┘
                          ▼
              DetectRes / SegmentRes / ...
```

**静态模型加速**：前三步在初始化时被捕获为 **CUDA Graph**，推理时仅需更新参数 → `launch()`。

---

## 六、各模型输出张量结构

推理时 engine 已包含 NMS 插件，输出的是**结构化最终结果**：

### Detect（目标检测）
| 张量索引 | 名称 | Shape | 说明 |
|----------|------|-------|------|
| 0 | input | `[B, 3, H, W]` | 输入图像 |
| 1 | num_dets | `[B, 1]` | 每张图检测到的目标数 |
| 2 | boxes | `[B, max_dets, 4]` | `[left, top, right, bottom]` |
| 3 | scores | `[B, max_dets]` | 置信度 |
| 4 | classes | `[B, max_dets]` | 类别 ID |

### Segment（实例分割）
| 张量索引 | 名称 | Shape | 说明 |
|----------|------|-------|------|
| 0 | input | `[B, 3, H, W]` | 输入 |
| 1 | num_dets | `[B, 1]` | 目标数 |
| 2 | boxes | `[B, max_dets, 4]` | 框坐标 |
| 3 | scores | `[B, max_dets]` | 置信度 |
| 4 | classes | `[B, max_dets]` | 类别 |
| 5 | masks | `[B, max_dets, H, W]` | 每个实例的掩码 |

### Pose（姿态估计）
| 张量索引 | 名称 | Shape | 说明 |
|----------|------|-------|------|
| 0 | input | `[B, 3, H, W]` | 输入 |
| 1 | num_dets | `[B, 1]` | 目标数 |
| 2 | boxes | `[B, max_dets, 4]` | 框坐标 |
| 3 | scores | `[B, max_dets]` | 置信度 |
| 4 | classes | `[B, max_dets]` | 类别 |
| 5 | kpts | `[B, max_dets, n_kpt, 2或3]` | 关键点 `[x, y, (conf)]` |

### OBB（旋转框检测）
| 张量索引 | 名称 | Shape | 说明 |
|----------|------|-------|------|
| 0 | input | `[B, 3, H, W]` | 输入 |
| 1 | num_dets | `[B, 1]` | 目标数 |
| 2 | boxes | `[B, max_dets, 5]` | `[left, top, right, bottom, theta]` |
| 3 | scores | `[B, max_dets]` | 置信度 |
| 4 | classes | `[B, max_dets]` | 类别 |

### Classify（分类）
| 张量索引 | 名称 | Shape | 说明 |
|----------|------|-------|------|
| 0 | input | `[B, 3, H, W]` | 输入 |
| 1 | topk | `[B, topk, 2]` | `[score, class_id]` |

---

## 七、使用指南

### 1. 环境准备

**前置依赖：**
- CUDA ≥ 11.0
- TensorRT ≥ 8.6
- CMake ≥ 3.18
- Linux (x86_64 或 ARM) 或 Windows

**编译安装：**

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
pip install "pybind11[global]"

# C++ 库编译
cmake -S . -B build \
  -D TRT_PATH=/usr/local/tensorrt \
  -D BUILD_PYTHON=ON \
  -D CMAKE_INSTALL_PREFIX=./install
cmake --build build -j$(nproc) --config Release --target install

# Python 包安装
pip install --upgrade build
python -m build --wheel
pip install dist/trtyolo-6.*-py3-none-any.whl
```

### 2. 模型转换（ONNX → TensorRT Engine）

使用配套工具 `trtyolo-export`：

```bash
pip install trtyolo-export
# 将 PyTorch/ONNX 模型转换为 TensorRT engine（自动嵌入 NMS 插件）
trtyolo-export --model yolov8n.pt --task detect --format engine
```

输出文件如 `yolo11n-with-plugin.engine`，其中 "with-plugin" 表示已包含 NMS 插件。

### 3. Python 推理

```python
import cv2
from trtyolo import TRTYOLO

# ---------- 初始化 ----------
model = TRTYOLO(
    "yolo11n-with-plugin.engine",
    task="detect",         # detect / segment / classify / pose / obb
    device=0,              # GPU ID
    swap_rb=True,          # BGR→RGB
    profile=True,          # 开启性能统计
    # mean=(0.485, 0.456, 0.406),   # 自定义归一化（可选）
    # std=(0.229, 0.224, 0.225),
    # border_value=0,                # 自定义填充值
    # input_size=(640, 640),         # 固定输入尺寸
)

# ---------- 单图推理 ----------
image = cv2.imread("test.jpg")
result = model.predict(image)  # 返回 sv.Detections
print(result.xyxy)             # (n, 4) 框坐标
print(result.class_id)         # (n,) 类别
print(result.confidence)       # (n,) 置信度

# ---------- 批量推理 ----------
images = [cv2.imread(f) for f in ["a.jpg", "b.jpg", "c.jpg"]]
results = model.predict(images)  # list[sv.Detections]，自动按batch分组

# ---------- 路径输入 ----------
result = model.predict("test.jpg")
results = model.predict(["a.jpg", "b.jpg"])

# ---------- 多线程（clone） ----------
import threading
model2 = model.clone()  # 共享 engine，独立 context
t = threading.Thread(target=lambda: model2.predict(img))
t.start()

# ---------- 性能报告 ----------
throughput, cpu_lat, gpu_lat = model.profile()
# throughput:  "Throughput: 120.14 qps"
# cpu_latency: "CPU Latency: min = 8.32 ms, max = 8.35 ms, mean = 8.33 ms, ..."
# gpu_latency: "GPU Latency: min = 8.12 ms, max = 8.15 ms, mean = 8.13 ms, ..."
```

**各任务结果类型（supervision 格式）：**

| task | predict 返回 |
|------|-------------|
| `detect` | `sv.Detections` |
| `segment` | `sv.Detections`（含 `.mask` 属性） |
| `classify` | `sv.Classifications` |
| `pose` | `sv.KeyPoints` |
| `obb` | `sv.Detections`（含 `ORIENTED_BOX_COORDINATES`） |

### 4. C++ 推理

```cpp
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    // ---------- 配置 ----------
    trtyolo::InferOption option;
    option.enableSwapRB();
    // option.setNormalizeParams({0.485, 0.456, 0.406}, {0.229, 0.224, 0.225});
    // option.setBorderValue(0.0f);
    // option.setInputDimensions(640, 640);
    // option.enablePerformanceReport();

    // ---------- 创建模型 ----------
    auto detector = std::make_unique<trtyolo::DetectModel>(
        "yolo11n-with-plugin.engine", option);

    // ---------- 加载图像 ----------
    cv::Mat img = cv::imread("test.jpg");
    trtyolo::Image input(img.data, img.cols, img.rows);

    // ---------- 推理 ----------
    trtyolo::DetectRes result = detector->predict(input);

    // 使用结果
    for (int i = 0; i < result.num; i++) {
        auto& box = result.boxes[i];
        std::cout << "class=" << result.classes[i]
                  << " score=" << result.scores[i]
                  << " box=[" << box.left << "," << box.top
                  << "," << box.right << "," << box.bottom << "]"
                  << std::endl;
    }

    // ---------- 批量推理 ----------
    std::vector<trtyolo::Image> images = {input, input};
    std::vector<trtyolo::DetectRes> results = detector->predict(images);

    // ---------- Clone ----------
    auto cloned = detector->clone();
    auto result2 = cloned->predict(input);
}
```

**C++ 编译链接（CMake）：**

```cmake
find_package(tensorrt-yolo REQUIRED)
target_link_libraries(your_app PRIVATE trtyolo::trtyolo)
```

### 5. Docker 部署

```bash
docker build -t tensorrt-yolo .
docker run --gpus all -v /path/to/models:/models tensorrt-yolo
```

---

## 八、关键设计模式

| 模式 | 应用位置 | 说明 |
|------|----------|------|
| **Pimpl（指针实现）** | `InferOption::Impl`, `BaseModel::Impl` | 隐藏实现细节，稳定 ABI，C++ 头文件无需暴露内部依赖 |
| **Strategy（策略模式）** | `BufferFactory` + 4种 `BaseBuffer` 子类 | 根据平台/配置自动切换内存策略 |
| **Template Method** | `withPerformanceReport()` | 装饰器：包裹推理+后处理，统一添加计时逻辑 |
| **Prototype（原型模式）** | `clone()` 方法 | 共享 engine，创建新 context，支持多线程 |
| **Object Pool** | `TRTManager::clone()` | engine 用 shared_ptr 共享，避免重复加载 |

---

## 九、性能优化要点总结

1. **CUDA Graph** — 静态模型将 H2D + Letterbox + enqueueV3 + D2H 一次捕获，消除 kernel launch 开销
2. **GPU Letterbox** — 前处理全部在 GPU 上完成，避免 GPU→CPU→GPU 的数据搬运
3. **NMS Plugin** — 后处理 NMS 在 GPU 上完成，结果直接写入 D2H 可读的状态
4. **Zero Copy (Jetson)** — 集成显卡使用 MappedBuffer，消除显式拷贝
5. **多 Context** — 每个 `clone()` 创建独立的 `IExecutionContext`，多流并行推理

---

## 十、源码逐行深入

### 10.1 TrtBackend — 推理后端的完整职责

`TrtBackend` 是推理流水线的**调度中心**，自己不计算，但把所有步骤串起来。

成员变量（`backend.hpp:64-87`）：

```cpp
class TrtBackend {
public:
    cudaStream_t            stream;        // CUDA 流
    InferConfig             infer_config;  // 推理配置
    std::vector<TensorInfo> tensor_infos;  // 所有 I/O 张量
    std::vector<Transform>  transforms;    // 仿射变换（每图一个）
    int4                    min_shape;     // 最小输入形状
    int4                    max_shape;     // 最大输入形状
    bool                    dynamic;       // 是否动态形状

private:
    unique_ptr<TRTManager> manager_;       // TensorRT engine + context
    CudaGraph              cuda_graph_;    // CUDA Graph
    unique_ptr<BaseBuffer> inputs_buffer_; // 输入 buffer
    BufferType             buffer_type_;   // 内存策略
    bool                   zero_copy_;     // Jetson Zero Copy
    int                    input_size_;    // 单图输入字节数
    int                    infer_size_;    // 单图推理输出字节数
};
```

**构造函数做的事（`backend.cpp:18-43`）：**

```
TrtBackend(engine_file, infer_config):
  1. cudaSetDevice(device_id)              ← 切到指定 GPU
  2. cudaStreamCreate(&stream)             ← 创建 CUDA 流
  3. SupportsIntegratedZeroCopy()          ← 检测是否 Jetson
  4. TRTManager() → ReadBinaryFromFile()   ← 读取 .engine 二进制
  5. manager_->initialize(blob, size)      ← 反序列化 engine，创建 context
  6. getTensorInfo()                       ← 遍历所有 I/O 张量
  7. initialize()                          ← 分配 buffer，初始化 Transform
  8. if (!dynamic) captureCudaGraph()      ← 静态模型：捕获 CUDA Graph
```

**推理调度 `infer()`（`backend.cpp:428-436`）：**

```cpp
void TrtBackend::infer(const std::vector<Image>& inputs) {
    cudaSetDevice(infer_config.device_id);  // 推理前切换设备
    if (dynamic) {
        dynamicInfer(inputs);   // 动态形状 → 逐步执行
    } else {
        staticInfer(inputs);    // 静态形状 → CUDA Graph 一键执行
    }
}
```

**为什么叫 Backend？** 因为它是底层实现，和前端接口解耦：

```
前端（你调用的）             后端（干活的）
─────────────────          ─────────────────
DetectModel                →  TrtBackend
  .predict(image)                 .infer(images)
                                  ├─ cudaLetterbox()      预处理 (GPU)
                                  ├─ manager_->enqueueV3()  推理 (GPU)
                                  └─ deviceToHost()       结果搬回 CPU
```

打个比方：DetectModel 是餐厅服务员（接单、上菜），TrtBackend 是后厨（洗菜、炒菜、装盘），TensorRT engine 是灶台（真正加热）。

---

### 10.2 CUDA Graph — GPU 指令的"录制 + 播放"

正常调用 CUDA kernel 时，每次都要经过 CPU → GPU 的**提交开销**：

```
每次推理（传统方式）：
  CPU → GPU: "跑 letterbox"    → 提交开销 ~10µs
  CPU → GPU: "跑 enqueueV3"    → 提交开销 ~10µs
  CPU → GPU: "结果拷回来"      → 提交开销 ~10µs
  总计：3 次提交，~30µs 白白浪费
```

CUDA Graph 的思路：**录制一次，反复播放**。把所有操作打包成一张"指令磁带"。

```
初始化（只做一次）：
  Graph = 录制[letterbox → enqueueV3 → memcpy]  ← 几十毫秒，但只做一次

每次推理：
  Graph.launch()  →  一次性提交全部，~3µs  ← 快了 10 倍
```

**在项目 `backend.cpp` 中的录制（captureCudaGraph）：**

```cpp
// Step 1: 预执行一次 warm up
manager_->enqueueV3(stream);
cudaStreamSynchronize(stream);

// Step 2: 开始录制
cuda_graph_.beginCapture(stream);      // "开始录像"
  inputs_buffer_->hostToDevice(stream); // H2D 拷贝
  cudaLetterbox(...);                  // 前处理 kernel
  manager_->enqueueV3(stream);         // TensorRT 推理（含 NMS）
  for (auto& t : tensor_infos)         // D2H 拷贝所有输出
      t.buffer->deviceToHost(stream);
cuda_graph_.endCapture(stream);        // "停止录像"

// Step 3: 获取图中各节点引用
cuda_graph_.initializeNodes(num);
```

**播放阶段（staticInfer）：**

```cpp
cuda_graph_.updateKernelNodeParams(0, ...);  // 更新 letterbox 参数
cuda_graph_.updateMemcpyNodeParams(0, ...);  // 更新拷贝参数
cuda_graph_.launch(stream);                  // "播放录像"，一键全部完成
```

**图中包含哪些节点？**

```
Node[0]:     H2D memcpy         (把新图片拷到 GPU)
Node[1..N]:  cudaLetterbox      (N = batch_size，每图一个)
Node[N+1]:   enqueueV3          (TensorRT 推理)
Node[N+2+]:  D2H memcpy         (每个输出张量拷贝回 CPU)
```

> **和 TaskFlow 的区别**：TaskFlow 在 CPU 上编排线程任务，CUDA Graph 在 GPU 上编排 kernel 指令。思想相似（预先定义图、消除调度开销），运行层次不同。

---

### 10.3 updateKernelNodeParams — 换参数不换图

CUDA Graph 的核心限制：**图一旦捕获，所有 kernel 参数指针就"冻住"了**。但每次推理图片的 GPU 地址不同。

`updateKernelNodeParams` 解决这个矛盾：**更新节点参数，无需重捕获**。

在 `staticInfer()` 中（`backend.cpp:228-246`）：

```cpp
for (int idx = 0; idx < num; ++idx) {
    // inputs[idx].ptr 指向新图片的 GPU 内存
    void* kernelParams[] = {
        (void*)&inputs[idx].ptr,          // ← 新图片指针     (变)
        (void*)&inputs[idx].width,        // ← 新图片宽度     (变)
        (void*)&inputs[idx].height,       // ← 新图片高度     (变)
        (void*)&inputs[idx].pitch,        // ← 新图片行距     (变)
        (void*)&infer_device_ptr,         // ← 输出位置       (不变)
        (void*)&max_shape.w,              // ← 模型输入宽     (不变)
        (void*)&max_shape.z,              // ← 模型输入高     (不变)
        (void*)&transforms.front().meta,  // ← 仿射变换元数据 (不变)
        (void*)&infer_config.config       // ← 预处理配置     (不变)
    };
    cuda_graph_.updateKernelNodeParams(idx, kernelParams);
}
```

**图解整个过程：**

```
初始化时（captureCudaGraph）：
  ┌──────────────────────────────────────────┐
  │          CUDA Graph（已冻住）              │
  │                                          │
  │  Node[0]: cudaLetterbox(图片A地址, ...)   │  ← 参数指向"图片A"
  │  Node[1]: enqueueV3(stream)              │
  │  Node[2]: deviceToHost                   │
  └──────────────────────────────────────────┘

每次推理时（staticInfer）：
  ┌──────────────────────────────────────────┐
  │  updateKernelNodeParams(0, 图片B的参数)   │  ← 换"图片B"指针
  │  cuda_graph_.launch(stream)              │  ← 直接执行，无需重捕获
  └──────────────────────────────────────────┘
```

**为什么这样设计？**

| 操作 | 耗时 |
|------|------|
| 捕获一次 CUDA Graph | 几十毫秒 |
| updateKernelNodeParams | 几乎零（改几个指针值） |

所以捕获一次，后面只换参数，这就是静态模型比动态模型快的核心原因。

打个比方：CUDA Graph 像已经填好地址的信封，`updateKernelNodeParams` 就是换一张信纸（改图片指针），不用重新写信封（重捕获）。

**底层实现（`core.cpp:185-200`）：**

```cpp
void CudaGraph::updateKernelNodeParams(size_t index, void** kernelParams) {
    // 1. 确认是 kernel 节点
    cudaKernelNodeParams nodeParams;
    // 2. 获取当前参数
    cudaGraphKernelNodeGetParams(nodes_[index], &nodeParams);
    // 3. 只换 kernelParams 指针
    nodeParams.kernelParams = kernelParams;
    // 4. 更新执行图节点
    cudaGraphExecKernelNodeSetParams(graphExec_, nodes_[index], &nodeParams);
}
```

---

### 10.4 cudaLetterbox — GPU 并行的图像预处理

YOLO 要求固定尺寸输入（如 640×640），但实际图片尺寸各异。`cudaLetterbox` 在 GPU 上完成等比缩放 + 填充 + 归一化。

**为什么放 GPU 而不是 CPU？**

```
CPU 方式（慢）：
  图片在 GPU → 拷回 CPU(慢) → OpenCV resize → 拷回 GPU(慢) → 推理
              ↑ 两次数据搬运是瓶颈 ↑

CUDA 方式（快）：
  图片在 GPU → cudaLetterbox → 直接推理
              ↑ 零搬运，GPU 原地搞定 ↑
```

**效果图：**

```
     输入图片 (1920×1080)           输出 (640×640)
  ┌─────────────────┐            ┌──────────────────┐
  │                 │            │  ░░░░░░░░░░░░░░  │ ← 灰色：填充区
  │   一辆车         │  ────→    │  ░░░░░░░░░░░░░░  │
  │                 │            │  ░░┌────────┐░░  │
  │                 │            │  ░░│  车    │░░  │ ← 等比缩放后居中
  └─────────────────┘            │  ░░└────────┘░░  │
                                 │  ░░░░░░░░░░░░░░  │
                                 └──────────────────┘
```

**每个像素并行做 4 步（`letterbox.cu`）：**

```
① 等比缩放   → 双线性插值，从源图 4 个邻居像素加权平均
② 通道交换   → BGR ↔ RGB（可选，SwapRB）
③ 归一化     → pixel * alpha + beta（默认 /255，可自定义 mean/std）
④ 边界填充   → 缩略图外区域填 border_value（默认 114）
```

精度：与 Python OpenCV 的 letterbox **像素误差为 0**。
