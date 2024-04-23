English | [中文](../cn/build_and_install.md)

# Quick Compilation and Installation

## Installing `tensorrt_yolo`

To install the `tensorrt_yolo` module from PyPI, simply execute the following command:

```bash
pip install -U tensorrt_yolo
```

If you wish to get the latest development version or contribute to the project, you can follow these steps to clone the code repository from GitHub and install:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  # Clone the code repository
cd TensorRT-YOLO
pip install --upgrade build
python -m build
pip install dist/tensorrt_yolo/tensorrt_yolo-3.*-py3-none-any.whl
```

In these steps, you can clone the code repository first, perform local builds, and then install the generated Wheel package using `pip`. This ensures that you install the latest version with the newest features and improvements.

## `Deploy` Compilation

### Requirements

- Linux: gcc/g++
- Windows: MSVC
- Xmake
- CUDA
- TensorRT

To meet deployment requirements, you can use Xmake for `Deploy` compilation. This process supports both dynamic and static library compilation:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
xmake f -k shared --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
# xmake f -k static --tensorrt="C:/Program Files/NVIDIA GPU Computing Toolkit/TensorRT/v8.6.1.6"
xmake -P . -r
```

During this process, you can use the xmake tool to choose between dynamic and static library compilation based on your deployment needs. You can also specify the TensorRT installation path to ensure correct linking of TensorRT libraries during compilation. Xmake automatically detects the CUDA installation path, but if you have multiple CUDA versions, you can specify them using `--cuda`. The compiled files will be located in the `lib` folder.