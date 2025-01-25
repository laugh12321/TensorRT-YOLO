[English](../en/build_and_install.md) | 简体中文

# 快速编译安装

## `TensorRT-YOLO` 编译

### 环境要求

- **Linux**: gcc/g++
- **Windows**: MSVC
- **构建工具**: CMake 或 Xmake
- **依赖库**: CUDA、cuDNN、TensorRT

> [!NOTE]  
> 如果您在 Windows 下进行开发，可以参考以下配置指南：
> 
> - [Windows 开发环境配置——NVIDIA 篇](https://www.cnblogs.com/laugh12321/p/17830096.html)  
> - [Windows 开发环境配置——C++ 篇](https://www.cnblogs.com/laugh12321/p/17827624.html)  

为了满足部署需求，您可以选择使用 Xmake 或 CMake 来编译动态库。以下是详细的编译指南：

首先，克隆 TensorRT-YOLO 仓库：

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  
cd TensorRT-YOLO
```

#### 使用 Xmake 编译

如果您选择使用 Xmake，可以按照以下步骤操作：

```bash
xmake f --tensorrt=/path/to/tensorrt --build_python=true
xmake -P . -r
```

#### 使用 CMake 编译

如果您选择使用 CMake，可以按照以下步骤操作：

```bash
pip install "pybind11[global]"
cmake -S . -B build -DTENSORRT_PATH=/usr/local/tensorrt -DBUILD_PYTHON=ON
cmake --build build -j$(nproc) --config Release
```

编译完成后，根目录下将生成一个名为 `lib` 的文件夹，同时在 `python/tensorrt_yolo/libs` 路径下生成相应的 Python 绑定。`lib` 文件夹包含以下内容：
- 名为 `deploy` 的动态库文件。
- 一个名为 `plugin` 的子文件夹，其中包含编译生成的 TensorRT 自定义插件动态库。

> [!NOTE]  
> 若不需要生成 Python 绑定，可以移除 `--build_python=true` 或 `-DBUILD_PYTHON=ON` 参数。

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
cd TensorRT-YOLO/python
pip install --upgrade build
python -m build --wheel
# 仅安装推理相关依赖
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl
# 安装模型导出及推理相关依赖
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl[export]
```
