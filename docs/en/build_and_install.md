English | [中文](../cn/build_and_install.md)

# Quick Compilation and Installation

## `TensorRT-YOLO` Compilation

### Environment Requirements

- Linux: gcc/g++
- Windows: MSVC
- CMake
- Xmake
- CUDA
- cuDNN
- TensorRT

> [!NOTE]  
> If you are developing on Windows, you can refer to the following configuration guides:
> 
> [Windows Development Environment Configuration — NVIDIA Chapter: Installing CUDA, cuDNN, and TensorRT](https://www.cnblogs.com/laugh12321/p/17830096.html) 
> 
> [Windows Development Environment Configuration — C++ Chapter: VSCode + MSVC/MinGW/Clangd/LLDB + Xmake](https://www.cnblogs.com/laugh12321/p/17827624.html) 

To meet deployment requirements, you can choose to compile the dynamic library using either Xmake or CMake. Here is a detailed compilation guide:

First, clone the TensorRT-YOLO repository:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO   
cd TensorRT-YOLO
```

#### Compiling with Xmake

If you choose to use Xmake, follow these steps:

```bash
xmake f --tensorrt="/usr/local/tensorrt"  # Configure the path to TensorRT
xmake -P . -r  # Compile the project
```

#### Compiling with CMake

If you choose to use CMake, follow these steps:

```bash
pip install "pybind11[global]"
mkdir build && cd build  # Create and enter the build directory
cmake -DTENSORRT_PATH=/usr/local/tensorrt ..  # Configure the path to TensorRT
cmake --build . -j8 --config Release  # Compile the project using 8 cores
```

After compilation, a folder named `lib` will be created in the root directory, along with the generation of corresponding Python bindings in the `tensorrt_yolo/libs` path. The `lib` folder contains two main elements: first, a dynamic library file named `deploy`, and second, a subfolder named `plugin`. Inside this `plugin` subfolder, you will find the TensorRT custom plugin dynamic library generated from the compilation.

## Installing `tensorrt_yolo`

If you only want to export ONNX models (with TensorRT plugins) for inference in the `tensorrt_yolo` project through the command-line interface (CLI) tool `trtyolo`, you can install via [PyPI](https://pypi.org/project/tensorrt-yolo) by simply running the following command:

```bash
pip install -U tensorrt_yolo
```

If you want to experience the same inference speed as C++, you will need to build the latest version of `tensorrt_yolo` yourself.

> [!NOTE]  
> Before building `tensorrt_yolo`, you need to compile `TensorRT-YOLO` first to generate the corresponding Python bindings.
> 

> [!IMPORTANT]   
> To avoid the `RuntimeError: Deploy initialization failed! Error: DLL load failed while importing pydeploy` error when building `tensorrt_yolo`, it is strongly recommended to follow these constraints:
>
> 1. Correctly install CUDA, cuDNN, TensorRT, and configure environment variables;
> 2. Ensure that the versions of cuDNN and TensorRT match the CUDA version;
> 3. Avoid having multiple versions of CUDA, cuDNN, and TensorRT.

```bash
cd TensorRT-YOLO
pip install --upgrade build
python -m build --wheel
# Install only the inference-related dependencies
pip install dist/tensorrt_yolo-4.*-py3-none-any.whl
# Install both the model export-related dependencies and the inference-related dependencies
pip install dist/tensorrt_yolo-4.*-py3-none-any.whl[export]
```
