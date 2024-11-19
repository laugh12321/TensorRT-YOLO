FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Required to build Ubuntu 22.04 without user prompts with DLFW container
ARG DEBIAN_FRONTEND=noninteractive

# Set environment and working directory
ENV TZ=Asia/Shanghai \
    PATH="${PATH}:/usr/local/tensorrt/bin" \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/tensorrt/lib"

# Install requried libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    gdb \
    libopencv-dev \
    wget \
    git \
    pkg-config \
    ssh \
    pbzip2 \
    bzip2 \
    unzip \
    axel

# Install python3
RUN apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      python3-dev \
      python3-wheel &&\
    cd /usr/local/bin &&\
    ln -s /usr/bin/python3 python &&\
    ln -s /usr/bin/pip3 pip;

# Install TensorRT
RUN axel --insecure -o /tmp/tensorrt.tar.gz https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.4.0/tars/TensorRT-10.4.0.26.Linux.x86_64-gnu.cuda-12.6.tar.gz \
    && tar -xvf /tmp/tensorrt.tar.gz -C /usr/local --transform 's/^TensorRT-10.4.0.26/tensorrt/' \
    && rm -f /tmp/tensorrt.tar.gz

# Install PyPI packages
RUN pip3 install --upgrade pip
RUN pip3 install setuptools>=41.0.0
RUN pip3 install "pybind11[global]"

# Install Cmake
RUN wget -O /tmp/cmake.sh https://www.ghproxy.cn/https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm /tmp/cmake.sh

WORKDIR /workspace
VOLUME /workspace

RUN ["/bin/bash"]