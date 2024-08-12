English | [简体中文](../cn/build_trt_custom_plugin.md)

# Build TensorRT Custom plugin

We need to build TensorRT OSS because the EfficientRotatedNMS plugin required by the OBB model is not included in the official TensorRT release.

# New Plugin

- [Efficient Rotated NMS Plugin](../../plugin/efficientRotatedNMSPlugin/README.md): This TensorRT plugin implements an efficient algorithm to perform Non Maximum Suppression for oriented bounding boxes object detection networks.

- [TRT_EfficientNMSX](https://github.com/levipereira/TensorRT/tree/release/8.6/plugin/efficientNMSPlugin): Similar to Efficient NMS, but returns the indices of the target boxes.

# Build

> [!NOTE]
> This example uses CUDA 11.8, cuDNN 8.9, and TensorRT 8.6 to demonstrate the TensorRT-OSS build process. Ensure that the downloaded TensorRT-OSS matches the TensorRT GA version you are using.

## Prerequisites

To build the TensorRT-OSS components, you will first need the following software packages.

**TensorRT GA Build**
* TensorRT v8.6.1.6
  * Download and extract the corresponding version of the TensorRT GA build from the [NVIDIA TensorRT 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download).

**System Packages**
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * Recommended versions:
    * cuda-11.8.0 + cuDNN-8.9
* [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
* [cmake](https://github.com/Kitware/CMake/releases) >= v3.13
* [python](<https://www.python.org/downloads/>) >= v3.8, <= v3.10.x
* [pip](https://pypi.org/project/pip/#history) >= v19.0
* Essential utilities
  * [git](https://git-scm.com/downloads), [pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/), [wget](https://www.gnu.org/software/wget/faq.html#download)

**Optional Packages**
* Containerized build
  * [Docker](https://docs.docker.com/install/) >= 19.03
  * [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* PyPI packages (for demo applications/tests)
  * [onnx](https://pypi.org/project/onnx/)
  * [onnxruntime](https://pypi.org/project/onnxruntime/)
  * [tensorflow-gpu](https://pypi.org/project/tensorflow/) >= 2.5.1
  * [Pillow](https://pypi.org/project/Pillow/) >= 9.0.1
  * [pycuda](https://pypi.org/project/pycuda/) < 2021.1
  * [numpy](https://pypi.org/project/numpy/)
  * [pytest](https://pypi.org/project/pytest/)
* Code formatting tools (for contributors)
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

  > NOTE: [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt), [cub](http://nvlabs.github.io/cub/), and [protobuf](https://github.com/protocolbuffers/protobuf.git) packages are downloaded along with TensorRT OSS, and not required to be installed.

## Downloading TensorRT Build

1. #### Download TensorRT OSS

    ```bash
    git clone -b release/8.6 https://github.com/nvidia/TensorRT TensorRT
    cd TensorRT
    git submodule update --init --recursive
    ```

2. #### (Optional - if not using TensorRT container) Specify the TensorRT GA release build path

    If using the TensorRT OSS build container, TensorRT libraries are preinstalled under `/usr/lib/x86_64-linux-gnu` and you may skip this step.

    Otherwise, download and extract the corresponding version of the TensorRT GA build from the [NVIDIA Developer community](https://developer.nvidia.com/tensorrt/download).

    **Example: Ubuntu 20.04 on x86-64 with cuda-11.8**

    ```bash
    cd ~/Downloads
    tar -xvzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
    export TRT_LIBPATH=`pwd`/TensorRT-8.6.1.6
    ```

    **Example: Windows on x86-64 with cuda-11.8**

    ```powershell
    Expand-Archive -Path TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip
    $env:TRT_LIBPATH="$pwd\TensorRT-8.6.1.6\lib"
    ```

3. #### Copy the `plugin/efficientRotatedNMSPlugin` Folder

    Copy the `plugin/efficientRotatedNMSPlugin` folder into the `plugin` directory within the TensorRT OSS.

4. #### Register the EfficientRotatedNMS Plugin

    In the `plugin/api/inferPlugin.cpp` file of TensorRT OSS, add the header file for the EfficientRotatedNMS plugin and initialize the plugin in the `initLibNvInferPlugins` function.

    ```cpp
    #include "efficientRotatedNMSPlugin/efficientRotatedNMSPlugin.h"

    // ...

    extern "C"
    {
        bool initLibNvInferPlugins(void* logger, const char* libNamespace)
        {
            // ...
            initializePlugin<nvinfer1::plugin::EfficientRotatedNMSPluginCreator>(logger, libNamespace);
            // ...
        }
    }
    ```

5. #### Add to TensorRT OSS `plugin/CMakeLists.txt`

    ```cmake
    set(PLUGIN_LISTS
        <!-- ... -->
        efficientRotatedNMSPlugin
        <!-- ... -->
    )
    ```

## Setting Up The Build Environment

For Linux platforms, we recommend that you generate a docker container for building TensorRT OSS as described below. For native builds, please install the [prerequisite](#prerequisites) *System Packages*.

1. #### Generate the TensorRT-OSS build container.
    The TensorRT-OSS build container can be generated using the supplied Dockerfiles and build scripts. The build containers are configured for building TensorRT OSS out-of-the-box.

    **Example: Ubuntu 20.04 on x86-64 with using cuda-11.8**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda11.8 --cuda 11.8.0
    ```
    **Example: Rockylinux8 on x86-64 with using cuda-11.8**
    ```bash
    ./docker/build.sh --file docker/rockylinux8.Dockerfile --tag tensorrt-rockylinux8-cuda11.8 --cuda 11.8.0
    ```
    **Example: Ubuntu 22.04 cross-compile for Jetson (aarch64) with cuda-11.4.2 (JetPack SDK)**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda11.4
    ```
    **Example: Ubuntu 22.04 on aarch64 with cuda-11.8**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu20.04-cuda11.8 --cuda 11.8.0
    ```

2. #### Launch the TensorRT-OSS build container.
    **Example: Ubuntu 20.04 build container**
    ```bash
    ./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda11.8 --gpus all
	```
	> NOTE:
  <br> 1. Use the `--tag` corresponding to build container generated in Step 1.
  <br> 2. [NVIDIA Container Toolkit](#prerequisites) is required for GPU access (running TensorRT applications) inside the build container.
  <br> 3. `sudo` password for Ubuntu build containers is 'nvidia'.
  <br> 4. Specify port number using `--jupyter <port>` for launching Jupyter notebooks.

## Building TensorRT-OSS
* Generate Makefiles and build.

    **Example: Linux (x86-64) build with cuda-11.8**
    ```bash
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=11.8
    make -j$(nproc)
    ```
    **Example: Linux (aarch64) build with default cuda-11.8**
    ```bash
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=11.8 -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
    make -j$(nproc)
    ```
    **Example: Native build on Jetson (aarch64) with cuda-11.4**
    ```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=11.4
  CC=/usr/bin/gcc make -j$(nproc)
    ```
  > NOTE: C compiler must be explicitly specified via CC= for native aarch64 builds of protobuf.

    **Example: Ubuntu 22.04 Cross-Compile for Jetson (aarch64) with cuda-11.4 (JetPack)**
    ```bash
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64.toolchain -DCUDA_VERSION=11.4 -DCUDNN_LIB=/pdk_files/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so -DCUBLAS_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublas.so -DCUBLASLT_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublasLt.so -DTRT_LIB_DIR=/pdk_files/tensorrt/lib
    make -j$(nproc)
    ```

    **Example: Native builds on Windows (x86) with cuda-11.8**
    ```powershell
    cd $TRT_OSSPATH
    mkdir -p build
    cd build
    cmake .. -DTRT_LIB_DIR="$env:TRT_LIBPATH" -DCUDNN_ROOT_DIR="$env:CUDNN_PATH" -DTRT_OUT_DIR="$pwd\out" msbuild TensorRT.sln /property:Configuration=Release -DCUDA_VERSION="11.8" -DCUDNN_VERSION="8.9" -DCMAKE_BUILD_TYPE=Release
    ```

	> NOTE:
	<br> 1. The default CUDA version used by CMake is 12.0.1. To override this, for example to 11.8, append `-DCUDA_VERSION=11.8` to the cmake command.
* Required CMake build arguments are:
	- `TRT_LIB_DIR`: Path to the TensorRT installation directory containing libraries.
	- `TRT_OUT_DIR`: Output directory where generated build artifacts will be copied.
* Optional CMake build arguments:
	- `CMAKE_BUILD_TYPE`: Specify if binaries generated are for release or debug (contain debug symbols). Values consists of [`Release`] | `Debug`
	- `CUDA_VERSION`: The version of CUDA to target, for example [`11.7.1`].
	- `CUDNN_VERSION`: The version of cuDNN to target, for example [`8.6`].
	- `PROTOBUF_VERSION`:  The version of Protobuf to use, for example [`3.0.0`]. Note: Changing this will not configure CMake to use a system version of Protobuf, it will configure CMake to download and try building that version.
	- `CMAKE_TOOLCHAIN_FILE`: The path to a toolchain file for cross compilation.
	- `BUILD_PARSERS`: Specify if the parsers should be built, for example [`ON`] | `OFF`.  If turned OFF, CMake will try to find precompiled versions of the parser libraries to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
	- `BUILD_PLUGINS`: Specify if the plugins should be built, for example [`ON`] | `OFF`. If turned OFF, CMake will try to find a precompiled version of the plugin library to use in compiling samples. First in `${TRT_LIB_DIR}`, then on the system. If the build type is Debug, then it will prefer debug builds of the libraries before release versions if available.
	- `BUILD_SAMPLES`: Specify if the samples should be built, for example [`ON`] | `OFF`.
	- `GPU_ARCHS`: GPU (SM) architectures to target. By default we generate CUDA code for all major SMs. Specific SM versions can be specified here as a quoted space-separated list to reduce compilation time and binary size. Table of compute capabilities of NVIDIA GPUs can be found [here](https://developer.nvidia.com/cuda-gpus). Examples:
        - NVidia A100: `-DGPU_ARCHS="80"`
        - Tesla T4, GeForce RTX 2080: `-DGPU_ARCHS="75"`
        - Titan V, Tesla V100: `-DGPU_ARCHS="70"`
        - Multiple SMs: `-DGPU_ARCHS="80 75"`
	- `TRT_PLATFORM_ID`: Bare-metal build (unlike containerized cross-compilation). Currently supported options: `x86_64` (default).

# References

- [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
- [levipereira/TensorRT](https://github.com/levipereira/TensorRT)

# Issue Reporting

If you encounter any issues while building TensorRT-OSS, please visit [NVIDIA/TensorRT issues](https://github.com/NVIDIA/TensorRT/issues) to report them!
