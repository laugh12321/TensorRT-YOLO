# 第一阶段：从 Python 镜像中获取 Python 3.12.9 环境
FROM python:3.12.9-slim-bookworm AS python_stage

# 第二阶段：创建最终的 Triton 镜像
FROM nvcr.io/nvidia/tritonserver:24.12-py3-min

# 元数据
LABEL maintainer="laugh12321@vip.qq.com"
LABEL description="🚀 Easier & Faster YOLO Deployment Toolkit for NVIDIA 🛠️"
LABEL version="6.3.0"

# 从 Python 镜像中复制核心 Python 文件
COPY --from=python_stage /usr/local/bin/python3.12 /usr/local/bin/python3.12
COPY --from=python_stage /usr/local/bin/pip3.12 /usr/local/bin/pip3.12
COPY --from=python_stage /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=python_stage /usr/local/include/python3.12 /usr/local/include/python3.12
COPY --from=python_stage /usr/local/lib/libpython3.12.so.1.0 /usr/local/lib/

# 合并多个操作到单个 RUN 指令
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip && \
    echo "/usr/local/lib" > /etc/ld.so.conf.d/python3.12.conf && \
    ldconfig && \
    # 添加 apt 安装
    apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libopencv-dev && \
    # 安装 CMake
    wget -O /tmp/cmake.sh https://www.ghproxy.cn/https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5-linux-x86_64.sh && \
    chmod +x /tmp/cmake.sh && \
    /tmp/cmake.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm /tmp/cmake.sh && \
    # 清理
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置环境变量
ENV PATH="/usr/local/bin:${PATH}" \
    PYTHONPATH="/usr/local/lib/python3.12/site-packages" \
    LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# 安装 Python 依赖并清理缓存
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip3 install --upgrade pip && \
    pip3 install torch==2.3.1 torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cpu && \
    pip3 install "pybind11[global]" && \
    pip3 cache purge && \
    rm -rf /root/.cache/pip

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print(torch.__version__)" || exit 1

WORKDIR /workspace
VOLUME /workspace

# 设置默认命令
CMD ["/bin/bash"]
