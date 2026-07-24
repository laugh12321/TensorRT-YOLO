# TensorRT-YOLO 使用指南

> 从环境搭建到成功推理的完整流程。
先nano禁用掉更新
dys@dys-desktop:~$ sudo apt-mark hold nvidia-l4t-kernel nvidia-l4t-kernel-dtbs nvidia-l4t-initrd
nvidia-l4t-kernel set on hold.
nvidia-l4t-kernel-dtbs set on hold.
nvidia-l4t-initrd set on hold.
dys@dys-desktop:~$ sudo apt-mark hold nvidia-l4t-core nvidia-jetpack
nvidia-l4t-core set on hold.
nvidia-jetpack set on hold.

dys@dys-desktop:~$ sudo systemctl mask update-notifier.service
sudo systemctl mask update-notifier-motd.service
---

## 一、环境要求

| 组件 | 版本 | 说明 |
|------|------|------|
| CUDA | ≥ 11.0 | NVIDIA GPU 驱动 |
| TensorRT | ≥ 8.6，推荐 10.x | `trtexec` 工具由它提供 |
| CMake | ≥ 3.18 | 构建工具 |
| OpenCV | 任意 | C++ 示例读图用 |
| Python | ≥ 3.8 | Python 推理 / 模型转换用 |

---

## 二、安装

### 2.1 克隆仓库

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
```

### 2.2 编译安装 TensorRT-YOLO 库（只需一次）

```bash
cmake -S . -B build \
  -D TRT_PATH=/your/tensorrt/path \
  -D BUILD_PYTHON=ON \
  -D CMAKE_INSTALL_PREFIX=./install

cmake --build build -j$(nproc) --config Release --target install
```

产物：

```
install/
├── include/               ← 头文件（trtyolo.hpp 等）
├── lib/
│   ├── libtrtyolo.so          ← C++ 推理库
│   └── libcustom_plugins.so   ← NMS 插件库
└── cmake/
    └── tensorrt-yolo-config.cmake  ← find_package 查找用
```

### 2.3 配置环境变量

让 CMake 能找到库，也让系统能找到动态库：

```bash
export CMAKE_PREFIX_PATH=/path/to/TensorRT-YOLO/install:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=/path/to/TensorRT-YOLO/install/lib:$LD_LIBRARY_PATH
```

### 2.4 安装 Python 相关包

```bash
# YOLO 官方工具（pt → onnx）
pip install ultralytics

# TensorRT-YOLO 配套工具（onnx → 带插件的 onnx）
pip install trtyolo-export

# Python 推理（可选，不写 Python 代码可以跳过）
pip install --upgrade build
cd TensorRT-YOLO
python -m build --wheel
pip install dist/trtyolo-6.*-py3-none-any.whl
```

---

## 三、模型转换

### 3.1 转换链路

```
yolo11n.pt  →  yolo11n.onnx  →  yolo11n-trtyolo.onnx  →  yolo11n.engine
  (PyTorch)      (通用格式)        (带NMS插件的ONNX)        (TensorRT引擎)
```

### 3.2 完整命令

```bash
# 步骤 1：pt → onnx（需要 ultralytics）
yolo export model=yolo11n.pt format=onnx batch=1

# 步骤 2：onnx → 带 NMS 插件的 onnx（需要 trtyolo-export）
trtyolo-export -i yolo11n.onnx -o yolo11n-trtyolo.onnx -s

# 步骤 3：onnx → TensorRT engine（需要系统装有 TensorRT）
trtexec --onnx=yolo11n-trtyolo.onnx --saveEngine=yolo11n.engine --fp16
```

> 步骤 3 中 `--fp16` 开启半精度推理，如果模型精度损失大可以去掉。
> 文件名叫什么 `.engine` 都行，代码只认文件内容，不认文件名。

### 3.3 其他 YOLO 变体

`ultralytics` 只支持 YOLOv3/v5/v8/v10/v11/v12/YOLO26/YOLO-World/YOLOE 等官方系列。

如果用其他团队的 YOLO（如美团 YOLOv6、旷视 YOLOX、百度 PP-YOLOE），需要用对应仓库的工具导出 ONNX，后面步骤 2、3 相同。

---

## 四、编译 Example

### 4.1 C++ Example

以 detect 为例：
cmake -S . -B build -D CMAKE_PREFIX_PATH=/home/dys/code/TensorRT-YOLO/install

```bash
cd TensorRT-YOLO/examples/detect

# 创建目录并放入模型和图片
mkdir -p models images
cp /your/path/yolo11n.engine models/
cp /your/path/test.jpg images/

# 编译
cmake -S . -B build
cmake --build build -j$(nproc)

# 运行（产物在 bin/ 下）
./bin/detect -e models/yolo11n.engine -i images/test.jpg -o output -l labels.txt
```

CMakeLists.txt 关键逻辑：

```cmake
find_package(OpenCV REQUIRED)           # 读图
find_package(TensorRT-YOLO REQUIRED)    # 推理库（第一步编译安装的）

add_executable(detect "detect.cpp")
target_link_libraries(detect PRIVATE
    ${OpenCV_LIBS}
    ${TensorRT-YOLO_LIBs}
)
```

### 4.2 Python Example

```bash
cd TensorRT-YOLO/examples/detect

python detect.py -e models/yolo11n.engine -i images -o output -l labels.txt
```

---

## 五、在自己的项目中使用

### 5.1 Python 最小示例

```python
import cv2
from trtyolo import TRTYOLO

# 初始化
model = TRTYOLO("yolo11n.engine", task="detect", swap_rb=True)

# 推理
image = cv2.imread("test.jpg")
result = model.predict(image)

# 结果
print(result.xyxy)         # (n, 4) 框坐标 [x1, y1, x2, y2]
print(result.class_id)     # (n,)  类别 ID
print(result.confidence)   # (n,)  置信度
```

### 5.2 C++ 最小示例

```cpp
#include <memory>
#include <opencv2/opencv.hpp>
#include "trtyolo.hpp"

int main() {
    // 配置
    trtyolo::InferOption option;
    option.enableSwapRB();

    // 创建模型
    auto detector = std::make_unique<trtyolo::DetectModel>(
        "yolo11n.engine", option);

    // 加载图片
    cv::Mat img = cv::imread("test.jpg");
    trtyolo::Image input(img.data, img.cols, img.rows);

    // 推理
    trtyolo::DetectRes result = detector->predict(input);

    // 使用结果
    for (int i = 0; i < result.num; i++) {
        auto& box = result.boxes[i];
        printf("class=%d score=%.2f box=[%.0f, %.0f, %.0f, %.0f]\n",
               result.classes[i], result.scores[i],
               box.left, box.top, box.right, box.bottom);
    }
}
```

CMakeLists.txt：

```cmake
find_package(OpenCV REQUIRED)
find_package(TensorRT-YOLO REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE ${OpenCV_LIBS} ${TensorRT-YOLO_LIBs})
```

---

## 六、支持的任务类型

| task | 模型类 | 用途 |
|------|--------|------|
| `detect` | `DetectModel` | 目标检测（框 + 类别） |
| `segment` | `SegmentModel` | 实例分割（框 + 类别 + 掩码） |
| `classify` | `ClassifyModel` | 图像分类（整图一个标签） |
| `pose` | `PoseModel` | 姿态估计（关键点） |
| `obb` | `OBBModel` | 旋转目标检测（旋转框 + 类别） |

---

## 七、常见问题

**Q: 模型文件名有要求吗？**

A: 没有，叫什么 `.engine` 都行，代码只读文件内容。

**Q: ultralytics 必须装特定版本吗？**

A: 不需要。ultralytics 和 TensorRT-YOLO 之间通过 ONNX 文件交互，没有版本耦合。装最新版即可。

**Q: 除了 ultralytics 的官方模型，其他 YOLO 能跑吗？**

A: 能。只要拿到 ONNX，就能用 `trtyolo-export` + `trtexec` 转成 engine，然后用 TensorRT-YOLO 推理。不同 YOLO 变体只是第一步导出 ONNX 的工具不同。

**Q: classify 和 detect 怎么选？**

A: classify 回答"整张图是什么"（一个标签），detect 回答"目标**在哪里**、是什么"（框 + 类别 + 位置）。需要知道目标位置就用 detect。

**Q: Jetson 设备上 `docker pull` 报 `no such host` 错误怎么办？**

错误示例：
```
Error response from daemon: failed to resolve reference "docker.io/ultralytics/ultralytics:latest-jetson-jetpack6":
dial tcp: lookup docker.mirrors.ustc.edu.cn on 127.0.0.53:53: no such host
```

A: 这是因为 Docker 镜像加速器地址失效或 DNS 无法解析。解决方法——更换稳定的镜像加速器。

编辑 Docker 配置文件：
```bash
sudo nano /etc/docker/daemon.json
```

将内容修改为（确保 JSON 格式正确，注意逗号）：
```json
{
  "registry-mirrors": [
    "https://docker.xuanyuan.me",
    "https://docker.1ms.run",
    "https://registry.docker-cn.com",
    "https://docker.m.daocloud.io",
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ]
}
```

保存后重启 Docker 服务：
```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

然后重新拉取镜像：
```bash
docker pull ultralytics/ultralytics:latest-jetson-jetpack6
```
