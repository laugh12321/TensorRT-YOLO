[English](../en/build_and_install.md) | 简体中文

# 快速编译安装

## `TensorRT-YOLO` 编译

### 环境要求

- Linux: gcc/g++
- Windows: MSVC
- CMake or Xmake
- CUDA
- cuDNN
- TensorRT

> [!NOTE]  
> 如果您在 Windows 下进行开发，可以参考以下配置指南：
> 
> [Windows 开发环境配置——NVIDIA 篇](https://www.cnblogs.com/laugh12321/p/17830096.html) 
> 
> [Windows 开发环境配置——C++ 篇](https://www.cnblogs.com/laugh12321/p/17827624.html) 

为了满足部署需求，您可以选择使用 Xmake 或 CMake 来编译动态库。以下是详细的编译指南：

首先，克隆 TensorRT-YOLO 仓库：

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  
cd TensorRT-YOLO
```

#### 使用 Xmake 编译

如果您选择使用 Xmake，可以按照以下步骤操作：

```bash
xmake f --tensorrt="/usr/local/tensorrt"  # 配置TensorRT路径
xmake -P . -r  # 编译项目
```

#### 使用 CMake 编译

如果您选择使用 CMake，可以按照以下步骤操作：

```bash
pip install "pybind11[global]"
mkdir build && cd build  # 创建并进入构建目录
cmake -DTENSORRT_PATH=/usr/local/tensorrt ..  # 配置TensorRT路径
cmake --build . -j8 --config Release  # 编译项目，使用8个核心
```

编译完成后，根目录下将创建一个名为 `lib` 的文件夹，同时在 `tensorrt_yolo/libs` 路径下生成相应的Python绑定。`lib` 文件夹内包含两个主要元素：首先是名为 `deploy` 的动态库文件，其次是一个名为 `plugin` 的子文件夹。在这个 `plugin` 子文件夹中，您会找到编译生成的 TensorRT 自定义插件动态库。

## 安装 `tensorrt_yolo`

如果您仅想通过 `tensorrt_yolo` 提供的命令行界面（CLI）工具 `trtyolo`，导出可供该项目推理的 ONNX 模型（带 TensorRT 插件），可以通过 [PyPI](https://pypi.org/project/tensorrt-yolo) 安装，只需执行以下命令即可：

```bash
pip install -U tensorrt_yolo
```

如果想体验与 C++ 同样的推理速度，则需要自行构建最新版本的 `tensorrt_yolo`。

> [!NOTE]  
> 在构建 `tensorrt_yolo` 前，需要先对 `TensorRT-YOLO` 进行编译，生成相应的Python绑定。
> 

> [!IMPORTANT]  
> 为了避免自行构建的 `tensorrt_yolo` 出现 `RuntimeError: Deploy initialization failed! Error: DLL load failed while importing pydeploy` 错误，强烈建议遵循以下约束：
>
> 1. 正确安装 CUDA、cuDNN、TensorRT 并配置环境变量；
> 2. 确保 cuDNN、TensorRT 版本与 CUDA 版本匹配；
> 3. 避免存在多个版本的 CUDA、cuDNN、TensorRT。

```bash
cd TensorRT-YOLO
pip install --upgrade build
python -m build --wheel
# 仅安装推理相关依赖
pip install dist/tensorrt_yolo-5.*-py3-none-any.whl
# 安装模型导出相关依赖以及推理相关依赖
pip install dist/tensorrt_yolo-5.*-py3-none-any.whl[export]
```
