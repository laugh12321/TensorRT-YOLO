[English](../en/build_and_install.md) | 简体中文

# 快速编译安装

## `TensorRT-YOLO` 编译

### 环境要求

- **Linux**: gcc/g++
- **Windows**: MSVC
- **构建工具**: CMake
- **依赖库**: CUDA、cuDNN、TensorRT

> [!NOTE]  
> 如果您在 Windows 下进行开发，可以参考以下配置指南：
>
> - [Windows 开发环境配置——NVIDIA 篇](https://www.cnblogs.com/laugh12321/p/17830096.html)
> - [Windows 开发环境配置——C++ 篇](https://www.cnblogs.com/laugh12321/p/17827624.html)

### 编译步骤

首先，克隆 TensorRT-YOLO 仓库：

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
```

然后使用 CMake，可以按照以下步骤操作：

```bash
pip install "pybind11[global]" # 安装 pybind11，用于生成 Python 绑定
cmake -S . -B build -D TRT_PATH=/your/tensorrt/dir -D BUILD_PYTHON=ON -D CMAKE_INSTALL_PREFIX=/your/tensorrt-yolo/install/dir
cmake --build build -j$(nproc) --config Release --target install
```

执行上述指令后，`tensorrt-yolo` 库将被安装到指定的 `CMAKE_INSTALL_PREFIX` 路径中。其中，`include` 文件夹中包含头文件，`lib` 文件夹中包含 `trtyolo` 动态库和 `custom_plugins` 动态库（仅在使用 `trtexec` 构建 OBB、Segment 或 Pose 模型时需要）。如果在编译时启用了 `BUILD_PYTHON` 选项，则还会在 `tensorrt_yolo/libs` 路径下生成相应的 Python 绑定文件。

> [!NOTE]  
> 在使用 C++ 版本的 `tensorrt-yolo` 库之前，请确保将指定的 `CMAKE_INSTALL_PREFIX` 路径添加到环境变量中，以便 CMake 的 `find_package` 能够找到 `tensorrt-yolo-config.cmake` 文件。可以通过以下命令完成此操作：
>
> ```bash
> export PATH=$PATH:/your/tensorrt-yolo/install/dir # linux
> $env:PATH = "$env:PATH;C:\your\tensorrt-yolo\install\dir;C:\your\tensorrt-yolo\install\dir\bin" # windows
> ```

## 安装 `tensorrt_yolo`

如果您仅想通过 `tensorrt_yolo` 提供的命令行工具 `trtyolo` 导出支持 TensorRT 插件推理的 ONNX 模型，可以通过 [PyPI](https://pypi.org/project/tensorrt-yolo) 直接安装：

```bash
pip install -U tensorrt_yolo
```

如果您希望体验与 C++ 相同的推理速度，则需要自行构建最新版本的 `tensorrt_yolo`。

> [!NOTE]  
> 在构建 `tensorrt_yolo` 前，需先编译 `TensorRT-YOLO` 并生成相应的 Python 绑定。

> [!IMPORTANT]  
> 为避免自行构建的 `tensorrt_yolo` 出现 `RuntimeError: Deploy initialization failed! Error: DLL load failed while importing pydeploy` 错误，请遵循以下约束：
>
> 1. 正确安装 CUDA、cuDNN、TensorRT 并配置环境变量；
> 2. 确保 cuDNN、TensorRT 版本与 CUDA 版本匹配；
> 3. 避免系统中存在多个版本的 CUDA、cuDNN、TensorRT。

```bash
pip install --upgrade build
python -m build --wheel
# 仅安装推理相关依赖
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl
# 安装模型导出及推理相关依赖
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl[export]
```
