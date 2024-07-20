English | [中文](../cn/build_and_install.md)

# Quick Build and Install

## `Deploy` Build

### Requirements

- Linux: gcc/g++
- Windows: MSVC
- Xmake
- CUDA
- cuDNN
- TensorRT

> [【Windows Development Environment Configuration—NVIDIA】Installing CUDA, cuDNN, and TensorRT](https://www.cnblogs.com/laugh12321/p/17830096.html)
>
> [【Windows Development Environment Configuration—C++】VSCode+MSVC/MinGW/Clangd/LLDB+Xmake](https://www.cnblogs.com/laugh12321/p/17827624.html)

To meet deployment needs, you can use Xmake for the `Deploy` build. This process supports compiling both dynamic and static libraries:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
xmake f -k shared --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
# xmake f -k static --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
xmake -P . -r
```

## Installing `tensorrt_yolo`

To install a version of the `tensorrt_yolo` module prior to version 4.0 via PyPI, simply run:

```bash
pip install -U tensorrt_yolo
```

If you want to experience inference speeds as fast as C++, you need to build the latest version of `tensorrt_yolo` yourself or download the pre-built Wheel package from the [Release](https://github.com/laugh12321/TensorRT-YOLO/releases) page and install it.

> Before building the `tensorrt_yolo` for the appropriate CUDA and TensorRT versions, you need to perform the `Deploy` build and then follow the steps below:
>
> Using `Python 3.10` as an example, if you need to compile for other Python versions, modify the `3.10` in the `xmake.lua` file under `add_requireconfs("pybind11.python", {version = "3.10", override = true})` to the corresponding version number.
```bash
conda create -n py10 python=3.10
conda activate py10
# After performing the Deploy build under py10, execute the following steps
pip install --upgrade build
python -m build --wheel
pip install dist/tensorrt_yolo-4.*-py3-none-any.whl
```

In these steps, you can clone the repository and build it locally, then use `pip` to install the generated Wheel package to ensure you are installing the latest version with the newest features and improvements.

During this process, you can use the xmake tool to choose between dynamic or static library compilation based on your deployment needs, and specify the TensorRT installation path to ensure proper linkage during compilation. Xmake will automatically recognize the CUDA installation path, and if you have multiple versions of CUDA, you can specify which one to use with `--cuda`. The compiled files will be located in the `lib` folder.