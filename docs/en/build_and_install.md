English | [中文](../cn/build_and_install.md)

# Quick Compilation and Installation

## Compiling `TensorRT-YOLO`

### Requirements

- **Linux**: gcc/g++
- **Windows**: MSVC
- **Build Tools**: CMake
- **Dependencies**: CUDA, cuDNN, TensorRT

> [!NOTE]  
> If you are developing on Windows, you can refer to the following setup guides:
>
> - [Windows Development Environment Setup – NVIDIA](https://www.cnblogs.com/laugh12321/p/17830096.html)
> - [Windows Development Environment Setup – C++](https://www.cnblogs.com/laugh12321/p/17827624.html)

### Compiling Steps

First, clone the TensorRT-YOLO repository:

```bash
git clone https://github.com/laugh12321/TensorRT-YOLO
cd TensorRT-YOLO
```

Then compile using CMake with the following steps:

```bash
pip install "pybind11[global]" # Install pybind11 to generate Python bindings
cmake -S . -B build -D TRT_PATH=/your/tensorrt/dir -D BUILD_PYTHON=ON -D CMAKE_INSTALL_PREFIX=/your/tensorrt-yolo/install/dir
cmake --build build -j$(nproc) --config Release --target install
```

After executing the above commands, the `tensorrt-yolo` library will be installed in the specified `CMAKE_INSTALL_PREFIX` directory. The `include` folder will contain the header files, and the `lib` folder will contain the `trtyolo` dynamic library and the `custom_plugins` dynamic library (only needed when building OBB, Segment, or Pose models with `trtexec`). If the `BUILD_PYTHON` option is enabled during compilation, the corresponding Python binding files will also be generated in the `tensorrt_yolo/libs` path.

> [!NOTE]  
> Before using the C++ version of the `tensorrt-yolo` library, ensure that the specified `CMAKE_INSTALL_PREFIX` path is added to the environment variables so that CMake's `find_package` can locate the `tensorrt-yolo-config.cmake` file. This can be done using the following command:
>
> ```bash
> export PATH=$PATH:/your/tensorrt-yolo/install/dir # linux
> $env:PATH = "$env:PATH;C:\your\tensorrt-yolo\install\dir;C:\your\tensorrt-yolo\install\dir\bin" # windows
> ```

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
