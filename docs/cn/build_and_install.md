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

> [!NOTE]  
> 如果你在 Windows 下进行开发，可以参考以下配置指南：
> 
> [【Windows 开发环境配置——NVIDIA 篇】CUDA、cuDNN、TensorRT 三件套安装](https://www.cnblogs.com/laugh12321/p/17830096.html)
> 
> [【Windows 开发环境配置——C++ 篇】VSCode+MSVC/MinGW/Clangd/LLDB+Xmake](https://www.cnblogs.com/laugh12321/p/17827624.html)

为了满足部署需求，您可以使用 Xmake 进行 `Deploy` 编译。此过程支持动态库和静态库的编译：

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
xmake f -k shared -m release --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
# xmake f -k static -m release --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
xmake -P . -r
```

## 安装 `tensorrt_yolo`

如果你仅想导出带 EfficientNMS TensorRT 插件的 ONNX 模型，可以通过 [PyPI](https://pypi.org/project/tensorrt-yolo) 安装 `4.0` 之前的版本，只需执行以下命令即可：

```bash
pip install -U tensorrt_yolo
```

如果想体验与 C++ 同样的推理速度，则需要自行构建最新版本的 `tensorrt_yolo`，或者通过 [Release](https://github.com/laugh12321/TensorRT-YOLO/releases) 下载构建好的 wheel 包安装。

> [!NOTE]  
> 在构建 `tensorrt_yolo` 前，需要先对 `Deploy` 进行编译。
> 
> 以 `Python 3.10` 为例，如果需要编译其他版本的 `Python`，请将 `xmake.lua` 文件中的 `add_requireconfs("pybind11.python", {version = "3.10", override = true})` 中的 `3.10` 修改为对应的版本号。

> [!IMPORTANT]  
> 为了避免自行构建的 `tensorrt_yolo` 出现 `RuntimeError: Deploy initialization failed! Error: DLL load failed while importing pydeploy` 错误，强烈建议遵循以下约束：
>
> 1. 正确安装 CUDA、cuDNN、TensorRT 并配置环境变量；
> 2. 确保 cuDNN、TensorRT 版本与 CUDA 版本匹配；
> 3. 避免存在多个版本的 CUDA、cuDNN、TensorRT；
> 4. 确保编译 `Deploy` 时的 Python 版本与 `xmake.lua` 中的设置以及 wheel 包安装环境的 Python 版本一致。

```bash
conda create -n py10 python=3.10
conda activate py10
# 在 py10 环境下对 Deploy 进行编译后执行以下步骤
pip install --upgrade build
python -m build --wheel
pip install dist/tensorrt_yolo-4.*-py3-none-any.whl
```

在以上步骤中，您可以先克隆代码库并进行本地构建，然后再使用 `pip` 安装生成的 Wheel 包，确保安装的是最新版本并具有最新的功能和改进。

在这个过程中，您可以使用 xmake 工具根据您的部署需求选择动态库或者静态库的编译方式，并且可以指定 TensorRT 的安装路径以确保编译过程中正确链接 TensorRT 库。Xmake 会自动识别 CUDA 的安装路径，如果您有多个版本的 CUDA，可以使用 `--cuda` 进行指定。编译后的文件将位于 `lib` 文件夹下。