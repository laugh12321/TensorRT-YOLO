[English](../en/build_and_install.md) | 简体中文

# 快速编译安装

## `Deploy` 编译

### 环境要求

- Linux: gcc/g++
- Windows: MSVC
- Xmake
- CUDA
- cuDNN
- TensorRT

> [【Windows 开发环境配置——NVIDIA 篇】CUDA、cuDNN、TensorRT 三件套安装](https://www.cnblogs.com/laugh12321/p/17830096.html)
> 
> [【Windows 开发环境配置——C++ 篇】VSCode+MSVC/MinGW/Clangd/LLDB+Xmake](https://www.cnblogs.com/laugh12321/p/17827624.html)

为了满足部署需求，您可以使用 Xmake 进行 `Deploy` 编译。此过程支持动态库和静态库的编译：

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
xmake f -k shared --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
# xmake f -k static --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
xmake -P . -r
```

## 安装 `tensorrt_yolo`

通过 PyPI 安装 `tensorrt_yolo` 模块的4.0之前的版本，您只需执行以下命令即可：

```bash
pip install -U tensorrt_yolo
```

如果需要体验与 C++ 一样快的推理速度则需要自行构建最新版本的 `tensorrt_yolo` 或者通过 [Release](https://github.com/laugh12321/TensorRT-YOLO/releases) 下载构建好的 Wheel 包安装。

> 在构建合适CUDA与TensorRT版本的 `tensorrt_yolo` 前，需要先对 `Deploy` 进行编译，然后再按照以下步骤构建：
> 
> 这里以 `Python 3.10` 为例，如果需要编译其他版本的 `Python`，请将 `xmake.lua` 文件中的 `add_requireconfs("pybind11.python", {version = "3.10", override = true})` 中的 `3.10` 修改为对应的版本号。
```bash
conda create -n py10 python=3.10
conda activate py10
# 在 py10 下对Deploy编译后执行以下步骤
pip install --upgrade build
python -m build --wheel
pip install dist/tensorrt_yolo-4.*-py3-none-any.whl
```

在以上步骤中，您可以先克隆代码库并进行本地构建，然后再使用 `pip` 安装生成的 Wheel 包，确保安装的是最新版本并具有最新的功能和改进。

在这个过程中，您可以使用 xmake 工具根据您的部署需求选择动态库或者静态库的编译方式，并且可以指定 TensorRT 的安装路径以确保编译过程中正确链接 TensorRT 库。Xmake 会自动识别 CUDA 的安装路径，如果您有多个版本的 CUDA，可以使用 `--cuda` 进行指定。编译后的文件将位于 `lib` 文件夹下。