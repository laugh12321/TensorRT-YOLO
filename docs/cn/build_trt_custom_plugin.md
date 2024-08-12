[English](../en/build_trt_custom_plugin.md) | 简体中文

# 构建 TensorRT 自定义插件

需要构建 TensorRT OSS，因为 OBB 模型所需的 EfficientRotatedNMS 插件不包含在官方发布的 TensorRT 中。

# 新插件

- [Efficient Rotated NMS Plugin](../../plugin/efficientRotatedNMSPlugin/README.md): 用于 OBB 目标检测网络的 NMS 插件。

# 构建

> [!NOTE]  
> 这里以 CUDA 11.8, cuDNN 8.9, TensorRT 8.6 为例介绍 TensorRT-OSS 的构建过程。实际构建时，请确保下载的 Tensor-OSS 与 TensorRT GA 版本一致。

## 先决条件

要构建 TensorRT-OSS 组件，首先需要以下软件包。

**TensorRT GA 构建**
* TensorRT v8.6.1.6
  * 请从 [NVIDIA TensorRT 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download) 下载并提取对应版本的 TensorRT GA 构建

**系统软件包**
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
  * 推荐版本：
    * cuda-11.8.0 + cuDNN-8.9
* [GNU make](https://ftp.gnu.org/gnu/make/) >= v4.1
* [cmake](https://github.com/Kitware/CMake/releases) >= v3.13
* [python](https://www.python.org/downloads/) >= v3.8，<= v3.10.x
* [pip](https://pypi.org/project/pip/#history) >= v19.0
* 必要工具
  * [git](https://git-scm.com/downloads)、[pkg-config](https://www.freedesktop.org/wiki/Software/pkg-config/)、[wget](https://www.gnu.org/software/wget/faq.html#download)

**可选软件包**
* 容器化构建
  * [Docker](https://docs.docker.com/install/) >= 19.03
  * [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker)
* PyPI 软件包（用于演示应用/测试）
  * [onnx](https://pypi.org/project/onnx/)
  * [onnxruntime](https://pypi.org/project/onnxruntime/)
  * [tensorflow-gpu](https://pypi.org/project/tensorflow/) >= 2.5.1
  * [Pillow](https://pypi.org/project/Pillow/) >= 9.0.1
  * [pycuda](https://pypi.org/project/pycuda/) < 2021.1
  * [numpy](https://pypi.org/project/numpy/)
  * [pytest](https://pypi.org/project/pytest/)
* 代码格式化工具（供贡献者使用）
  * [Clang-format](https://clang.llvm.org/docs/ClangFormat.html)
  * [Git-clang-format](https://github.com/llvm-mirror/clang/blob/master/tools/clang-format/git-clang-format)

  > 注意: [onnx-tensorrt](https://github.com/onnx/onnx-tensorrt)、[cub](http://nvlabs.github.io/cub/) 和 [protobuf](https://github.com/protocolbuffers/protobuf.git) 软件包会与 TensorRT OSS 一起下载，不需要单独安装。

## 下载 TensorRT Build

1. #### 下载 TensorRT OSS

    ```bash
	git clone -b release/8.6 https://github.com/nvidia/TensorRT TensorRT
	cd TensorRT
	git submodule update --init --recursive
	```

2. #### (可选 - 如果不使用 TensorRT 容器) 指定 TensorRT GA 发布构建路径

    如果使用 TensorRT OSS 构建容器，则 TensorRT 库预装在 `/usr/lib/x86_64-linux-gun` 您可以跳过此步骤。

    否则，请从 [NVIDIA 开发者社区](https://developer.nvidia.com/tensorrt/download) 下载并提取对应版本的 TensorRT GA 构建。

    **示例：基于 x86-64 架构的 Ubuntu 20.04，使用 cuda-11.8**

    ```bash
    cd ~/Downloads
    tar -xvzf TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz
    export TRT_LIBPATH=`pwd`/TensorRT-8.6.1.6
    ```

    **示例：基于 x86-64 架构的 Windows，使用 cuda-11.8**

    ```powershell
                            
    Expand-Archive -Path TensorRT-8.6.1.6.Windows10.x86_64.cuda-11.8.zip
    $env:TRT_LIBPATH="$pwd\TensorRT-8.6.1.6\lib"
    ```

3. #### 复制 `plugin/efficientRotatedNMSPlugin` 文件夹

    将 `plugin/efficientRotatedNMSPlugin` 文件夹复制到 TensorRT OSS 中 `plugin` 文件夹内。

4. #### 注册 EfficientRotatedNMS 插件

    在 TensorRT OSS 的 `plugin/api/inferPlugin.cpp` 文件中，添加 EfficientRotatedNMS 插件的头文件，并在 `initLibNvInferPlugins` 函数中初始化插件。

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

5. #### 添加至 TensorRT OSS `plugin/CMakeLists.txt`

    ```cmake
    set(PLUGIN_LISTS
        <!-- ... -->
        efficientRotatedNMSPlugin
        <!-- ... -->
    )
    ```

## 设置构建环境

对于 Linux 平台，建议按照以下步骤生成一个用于构建 TensorRT OSS 的 Docker 容器。如果是本地构建，请安装[先决条件](#prerequisites)中的*系统软件包*。

1. #### 生成 TensorRT-OSS 构建容器。
    可以使用提供的 Dockerfiles 和构建脚本生成 TensorRT-OSS 构建容器。这些构建容器已配置好，可以直接用于构建 TensorRT OSS。

    **示例：基于 x86-64 架构的 Ubuntu 20.04，使用 cuda-11.8**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt-ubuntu20.04-cuda11.8 --cuda 11.8.0
    ```
    **示例：基于 x86-64 架构的 Rockylinux8，使用 cuda-12.8**
    ```bash
    ./docker/build.sh --file docker/rockylinux8.Dockerfile --tag tensorrt-rockylinux8-cuda11.8 --cuda 11.8.0
    ```
    **示例：基于 Jetson（aarch64）架构的 Ubuntu 22.04 交叉编译，使用 cuda-11.4.2（JetPack SDK）**
    ```bash
    ./docker/build.sh --file docker/ubuntu-cross-aarch64.Dockerfile --tag tensorrt-jetpack-cuda11.4
    ```
    **示例：基于 aarch64 架构的 Ubuntu 22.04，使用 cuda-11.8**
    ```bash
    ./docker/build.sh --file docker/ubuntu-20.04-aarch64.Dockerfile --tag tensorrt-aarch64-ubuntu20.04-cuda11.8 --cuda 11.8.0
    ```

2. #### 启动 TensorRT-OSS 构建容器。
    **示例：Ubuntu 20.04 构建容器**
	```bash
	./docker/launch.sh --tag tensorrt-ubuntu20.04-cuda11.8 --gpus all
	```
	> 注意：
  <br> 1. 使用第1步中生成的构建容器对应的 `--tag`。
  <br> 2. 构建容器内部需要 [NVIDIA Container Toolkit](#prerequisites) 才能访问 GPU（运行 TensorRT 应用程序）。
  <br> 3. Ubuntu 构建容器的 `sudo` 密码是 'nvidia'。
  <br> 4. 使用 `--jupyter <port>` 指定端口号以启动 Jupyter notebooks。

## 构建 TensorRT-OSS
* 生成 Makefiles 并进行构建。

    **示例：基于 x86-64 架构的 Linux，使用默认的 cuda-11.8**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out
	make -j$(nproc)
	```
    **示例：基于 aarch64 架构的 Linux，使用默认的 cuda-11.8**
	```bash
	cd $TRT_OSSPATH
	mkdir -p build && cd build
	cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64-native.toolchain
	make -j$(nproc)
	```
    **示例：基于 Jetson（aarch64）架构的本地构建，使用 cuda-11.4**
	```bash
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DTRT_PLATFORM_ID=aarch64 -DCUDA_VERSION=11.4
    CC=/usr/bin/gcc make -j$(nproc)
	```
  > 注意：本地 aarch64 构建 protobuf 时，必须通过 CC= 显式指定 C 编译器。

    **示例：基于 Jetson（aarch64）架构的 Ubuntu 22.04 交叉编译，使用 cuda-11.4（JetPack）**
	```bash
    cd $TRT_OSSPATH
    mkdir -p build && cd build
    cmake .. -DCMAKE_TOOLCHAIN_FILE=$TRT_OSSPATH/cmake/toolchains/cmake_aarch64.toolchain -DCUDA_VERSION=11.4 -DCUDNN_LIB=/pdk_files/cudnn/usr/lib/aarch64-linux-gnu/libcudnn.so -DCUBLAS_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublas.so -DCUBLASLT_LIB=/usr/local/cuda-11.4/targets/aarch64-linux/lib/stubs/libcublasLt.so -DTRT_LIB_DIR=/pdk_files/tensorrt/lib

    make -j$(nproc)
	```

    **示例：基于 x86 架构的 Windows 本地构建，使用 cuda-11.8**
	```powershell
	cd $TRT_OSSPATH
	mkdir -p build
	cd -p build
    cmake .. -DTRT_LIB_DIR="$env:TRT_LIBPATH" -DCUDNN_ROOT_DIR="$env:CUDNN_PATH" -DTRT_OUT_DIR="$pwd\\out" msbuild TensorRT.sln /property:Configuration=Release -DCUDA_VERSION="11.8" -DCUDNN_VERSION="8.9" -DCMAKE_BUILD_TYPE=Release
	```

	> 注意：
	<br> 1. CMake 默认使用的 CUDA 版本是 12.0.1。要覆盖此版本，例如更改为 11.8，请在 cmake 命令后添加 `-DCUDA_VERSION=11.8`。
* 必需的 CMake 构建参数：
	- `TRT_LIB_DIR`: 包含库的 TensorRT 安装目录的路径。
	- `TRT_OUT_DIR`: 生成的构建工件将被复制到的输出目录。
* 可选的 CMake 构建参数：
	- `CMAKE_BUILD_TYPE`: 指定生成的二进制文件是用于发布还是调试（包含调试符号）。取值为 [`Release`] | `Debug`
	- `CUDA_VERSION`: 目标 CUDA 的版本，例如 [`11.7.1`].
	- `CUDNN_VERSION`: 目标 cuDNN 的版本，例如 [`8.6`].
	- `PROTOBUF_VERSION`:  使用的 Protobuf 版本，例如 [`3.0.0`]. 注意：更改此项不会配置 CMake 使用系统版本的 Protobuf，它将配置 CMake 下载并尝试构建该版本。
	- `CMAKE_TOOLCHAIN_FILE`: 交叉编译的工具链文件路径。
	- `BUILD_PARSERS`: 指定是否构建解析器，例如 [`ON`] | `OFF`. 如果关闭，CMake 将尝试找到预编译的解析器库，用于编译示例。首先在 `${TRT_LIB_DIR}` 中寻找，然后在系统中寻找。如果构建类型为调试，它将优先使用调试版本的库（如果可用），然后才是发布版本。
	- `BUILD_PLUGINS`: 指定是否构建插件，例如 [`ON`] | `OFF`. 如果关闭，CMake 将尝试找到预编译的插件库，用于编译示例。首先在 `${TRT_LIB_DIR}` 中寻找，然后在系统中寻找。如果构建类型为调试，它将优先使用调试版本的库（如果可用），然后才是发布版本。
	- `BUILD_SAMPLES`: 指定是否构建示例，例如 [`ON`] | `OFF`.
	- `GPU_ARCHS`: 目标 GPU（SM）架构。默认情况下，我们为所有主要的 SM 生成 CUDA 代码。可以在此处指定特定的 SM 版本，以减少编译时间和二进制文件大小。NVIDIA GPU 的计算能力表可以在[这里](https://developer.nvidia.com/cuda-gpus)找到。示例：
        - NVidia A100: `-DGPU_ARCHS="80"`
        - Tesla T4, GeForce RTX 2080: `-DGPU_ARCHS="75"`
        - Titan V, Tesla V100: `-DGPU_ARCHS="70"`
        - 多个 SM: `-DGPU_ARCHS="80 75"`
	- `TRT_PLATFORM_ID`: 裸机构建（与容器化交叉编译不同）。当前支持的选项：`x86_64`（默认）。

# 参考资料

- [NVIDIA/TensorRT](https://github.com/NVIDIA/TensorRT)
- [levipereira/TensorRT](https://github.com/levipereira/TensorRT)

# 问题反馈

如果在构建 TensorRT-OSS 时遇到问题，请访问 [NVIDIA/TensorRT issues](https://github.com/NVIDIA/TensorRT/issues) 进行反馈！
