# 第一阶段：基础镜像，安装必要的依赖项
FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04 AS base

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

RUN apt update && apt install -y --no-install-recommends \
    software-properties-common \
    liblapack-dev \
    libblas-dev \
    unzip \
    p7zip \
    git \
    wget \
    python3-pip \
    ffmpeg \
    libopencv-dev \
    && add-apt-repository ppa:xmake-io/xmake \
    && apt install -y --no-install-recommends xmake \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

# 第二阶段：安装 TensorRT
FROM base AS trt_install

RUN  wget --no-check-certificate -O tensorrt.tar.gz https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz \
    && tar -xvf tensorrt.tar.gz -C /usr/local --transform 's/^TensorRT-8.6.1.6/tensorrt/' \
    && rm -f tensorrt.tar.gz

# 第三阶段：最终镜像
FROM base

COPY --from=trt_install /usr/local/tensorrt /usr/local/tensorrt

# 安装 Python 包
RUN python3 -m pip install /usr/local/tensorrt/python/tensorrt-8.6.1-cp38-none-linux_x86_64.whl \
    /usr/local/tensorrt/python/tensorrt_dispatch-8.6.1-cp38-none-linux_x86_64.whl \
    /usr/local/tensorrt/python/tensorrt_lean-8.6.1-cp38-none-linux_x86_64.whl \
    /usr/local/tensorrt/onnx_graphsurgeon/onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl \
    cuda-python==11.6.1

RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116

RUN python3 -m pip install tensorrt-yolo

# 设置环境变量
ENV XMAKE_ROOT=y \
    PATH="${PATH}:/usr/local/tensorrt/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/tensorrt/lib"

# 默认命令
CMD ["/bin/bash"]
