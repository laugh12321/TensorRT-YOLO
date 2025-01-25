English | [中文](../cn/build_and_install.md)

# Quick Compilation and Installation

## Compiling `TensorRT-YOLO`

### Requirements

- **Linux**: gcc/g++
- **Windows**: MSVC
- **Build Tools**: CMake or Xmake
- **Dependencies**: CUDA, cuDNN, TensorRT

> [!NOTE]  
> If you are developing on Windows, you can refer to the following setup guides:
> 
> - [Windows Development Environment Setup – NVIDIA](https://www.cnblogs.com/laugh12321/p/17830096.html)  
> - [Windows Development Environment Setup – C++](https://www.cnblogs.com/laugh12321/p/17827624.html)  

To meet deployment requirements, you can choose to use Xmake or CMake to compile the dynamic library. Below are detailed build instructions:

First, clone the TensorRT-YOLO repository:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO  
cd TensorRT-YOLO
```

#### Building with Xmake

If you choose to use Xmake, follow these steps:

```bash
xmake f --tensorrt=/path/to/tensorrt --build_python=true
xmake -P . -r
```

#### Building with CMake

If you choose to use CMake, follow these steps:

```bash
pip install "pybind11[global]"
cmake -S . -B build -DTENSORRT_PATH=/usr/local/tensorrt -DBUILD_PYTHON=ON
cmake --build build -j$(nproc) --config Release
```

After compilation, a folder named `lib` will be created in the root directory, and the corresponding Python bindings will be generated under `python/tensorrt_yolo/libs`. The `lib` folder contains the following:
- A dynamic library file named `deploy`.
- A subfolder named `plugin`, which contains the compiled TensorRT custom plugin dynamic libraries.

> [!NOTE]  
> If Python bindings are not needed, you can remove the `--build_python=true` or `-DBUILD_PYTHON=ON` parameter.

## Installing `tensorrt_yolo`

If you only want to use the command-line tool `trtyolo` provided by `tensorrt_yolo` to export ONNX models with TensorRT plugins for inference, you can install it directly via [PyPI](https://pypi.org/project/tensorrt-yolo):

```bash
pip install -U tensorrt_yolo
```

If you want to experience the same inference speed as C++, you need to build the latest version of `tensorrt_yolo` yourself.

> [!NOTE]  
> Before building `tensorrt_yolo`, you must first compile `TensorRT-YOLO` and generate the corresponding Python bindings.

> [!IMPORTANT]  
> To avoid the `RuntimeError: Deploy initialization failed! Error: DLL load failed while importing pydeploy` error when building `tensorrt_yolo` yourself, please adhere to the following constraints:
>
> 1. Install CUDA, cuDNN, and TensorRT correctly and configure the environment variables.
> 2. Ensure that the cuDNN and TensorRT versions are compatible with the CUDA version.
> 3. Avoid having multiple versions of CUDA, cuDNN, or TensorRT installed.

```bash
cd TensorRT-YOLO/python
pip install --upgrade build
python -m build --wheel
# Install only inference-related dependencies
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl
# Install both model export and inference-related dependencies
pip install dist/tensorrt_yolo-6.*-py3-none-any.whl[export]
```
