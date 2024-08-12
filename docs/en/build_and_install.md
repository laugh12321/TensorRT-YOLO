English | [中文](../cn/build_and_install.md)

# Quick Build and Installation

> [!NOTE]  
> If you want to use the [EfficientRotatedNMS](../../plugin/efficientRotatedNMSPlugin) plugin for inference with OBB models, please refer to [Build TensorRT Custom plugin](./build_trt_custom_plugin.md) for the build process.

## `Deploy` Build

### Environment Requirements

- Linux: gcc/g++
- Windows: MSVC
- Xmake
- CUDA
- cuDNN
- TensorRT

> [!NOTE]  
> If you are developing on Windows, you can refer to the following configuration guides:
> 
> [【Windows Development Environment Configuration - NVIDIA】Installation of CUDA, cuDNN, and TensorRT](https://www.cnblogs.com/laugh12321/p/17830096.html)
> 
> [【Windows Development Environment Configuration - C++】VSCode+MSVC/MinGW/Clangd/LLDB+Xmake](https://www.cnblogs.com/laugh12321/p/17827624.html)

To meet deployment requirements, you can use Xmake to perform the `Deploy` build. This process supports the compilation of dynamic and static libraries:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
xmake f -k shared -m release --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
# xmake f -k static -m release --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
xmake -P . -r
```

## Installing `tensorrt_yolo`

If you only want to export an ONNX model with the EfficientNMS TensorRT plugin, you can install a version prior to `4.0` from [PyPI](https://pypi.org/project/tensorrt-yolo) by executing the following command:

```bash
pip install -U tensorrt_yolo
```

If you want to experience the same inference speed as in C++, you need to build the latest version of `tensorrt_yolo` yourself or download a pre-built wheel package from the [Release](https://github.com/laugh12321/TensorRT-YOLO/releases) page.

> [!NOTE]  
> Before building `tensorrt_yolo`, you need to compile the `Deploy`.
> 
> For example, with `Python 3.10`, if you need to compile for other Python versions, please modify the `version` in `add_requireconfs("pybind11.python", {version = "3.10", override = true})` in the `xmake.lua` file to the corresponding version number.

> [!IMPORTANT]  
> To avoid `RuntimeError: Deploy initialization failed! Error: DLL load failed while importing pydeploy` when building `tensorrt_yolo` yourself, it is strongly recommended to follow these constraints:
>
> 1. Correctly install CUDA, cuDNN, TensorRT, and configure environment variables;
> 2. Ensure the versions of cuDNN and TensorRT match the CUDA version;
> 3. Avoid having multiple versions of CUDA, cuDNN, and TensorRT;
> 4. Ensure the Python version used to compile `Deploy` matches the setting in `xmake.lua` and the Python version of the wheel package installation environment.

```bash
conda create -n py10 python=3.10
conda activate py10
# After compiling Deploy in the py10 environment, execute the following steps
pip install --upgrade build
python -m build --wheel
pip install dist/tensorrt_yolo-4.*-py3-none-any.whl
```

In the above steps, you can clone the code repository and build it locally, then use `pip` to install the generated wheel package to ensure you have the latest version with the newest features and improvements.

During this process, you can use the xmake tool to choose between dynamic or static library compilation according to your deployment needs, and specify the TensorRT installation path to ensure correct linkage during the build process. Xmake will automatically detect the CUDA installation path; if you have multiple versions of CUDA, you can specify it using `--cuda`. The compiled files will be located in the `lib` folder.