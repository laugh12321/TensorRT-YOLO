/**
 * @file core.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义了 TensorRT 日志记录器和 CUDA 图管理类的接口
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <atomic>
#include <memory>

#include "NvInferRuntime.h"

namespace deploy {

/**
 * @brief TensorRT 运行时管理器类，用于单例模式管理 TensorRT 的运行时环境和日志记录。
 *
 */
class TRTManager {
public:
    /**
     * @brief 获取 TensorRT 运行时对象。
     *
     * @return nvinfer1::IRuntime* 返回 TensorRT 的运行时实例。
     */
    static nvinfer1::IRuntime* getRuntime();

    /**
     * @brief 引擎删除器，用于安全释放 TensorRT 引擎资源。
     *
     * @param engine 要释放的 TensorRT 引擎对象。
     */
    static void engineDeleter(nvinfer1::ICudaEngine* engine);

private:
    TRTManager();
    ~TRTManager() = default;

    TRTManager(const TRTManager&)            = delete;
    TRTManager& operator=(const TRTManager&) = delete;

    struct TRTLogger;                             // < TensorRT 日志记录器
    std::unique_ptr<TRTLogger>          logger;   // < 日志记录器实例
    std::unique_ptr<nvinfer1::IRuntime> runtime;  // < TensorRT 运行时实例

    static TRTManager&      instance();           // < 获取单例实例
    static std::atomic<int> refCount;             // < 引用计数，用于管理运行时的释放时机
};

/**
 * @brief CUDA 图类，负责管理 CUDA 图的创建、执行、捕获等操作
 *
 * 该类封装了 CUDA 图的相关操作，包括捕获、执行、初始化节点、更新节点参数等。
 * CUDA 图是一种高效的 CUDA 操作执行模型，可以显著提高一些重复性操作的性能。
 */
class CudaGraph {
public:
    explicit CudaGraph() = default;
    ~CudaGraph() { destroy(); }
    CudaGraph(const CudaGraph&)            = delete;
    CudaGraph& operator=(const CudaGraph&) = delete;
    CudaGraph(CudaGraph&&)                 = delete;
    CudaGraph& operator=(CudaGraph&&)      = delete;

    /**
     * @brief 销毁 CUDA 图和执行图资源
     *
     * 销毁 `graphExec_` 和 `graph_`，释放与 CUDA 图相关的所有资源。
     */
    void destroy();

    /**
     * @brief 开始捕获 CUDA 图
     *
     * 开始在指定的流中捕获 CUDA 操作。捕获模式为 `cudaStreamCaptureModeThreadLocal`。
     *
     * @param stream 要捕获的 CUDA 流。
     */
    void beginCapture(cudaStream_t stream);

    /**
     * @brief 结束捕获 CUDA 图
     *
     * 完成捕获并实例化图执行句柄。
     *
     * @param stream 要结束捕获的 CUDA 流。
     */
    void endCapture(cudaStream_t stream);

    /**
     * @brief 启动 CUDA 图
     *
     * 在指定的流中执行 CUDA 图并同步流。
     *
     * @param stream 要启动 CUDA 图的流。
     */
    void launch(cudaStream_t stream);

    /**
     * @brief 初始化图中的节点
     *
     * 获取并初始化 CUDA 图中的节点。如果 `num` 为 0，则自动获取节点数。
     *
     * @param num 要初始化的节点数量。
     * @throws std::runtime_error 如果图中没有节点。
     */
    void initializeNodes(size_t num = 0);

    /**
     * @brief 更新内核节点的参数
     *
     * 更新指定内核节点的参数。
     *
     * @param index 节点的索引。
     * @param kernelParams 新的内核参数。
     *
     * @throws std::runtime_error 如果指定的节点不是内核节点。
     */
    void updateKernelNodeParams(size_t index, void** kernelParams);

    /**
     * @brief 更新 memcpy 节点的参数
     *
     * 更新指定 memcpy 节点的参数。
     *
     * @param index 节点的索引。
     * @param src 源内存地址。
     * @param dst 目标内存地址。
     * @param size 要复制的大小。
     *
     * @throws std::runtime_error 如果指定的节点不是 memcpy 节点。
     */
    void updateMemcpyNodeParams(size_t index, void* src, void* dst, size_t size);

private:
    /**
     * @brief 获取节点类型
     *
     * 获取指定索引的节点类型。如果节点类型无效，将抛出异常。
     *
     * @param index 节点索引。
     * @return cudaGraphNodeType 节点的类型。
     */
    cudaGraphNodeType getNodeType(size_t index);

private:
    std::unique_ptr<cudaGraphNode_t[]> nodes_;                // < 存储图节点的唯一指针
    cudaGraph_t                        graph_     = nullptr;  // < 图句柄，默认初始化为 nullptr
    cudaGraphExec_t                    graphExec_ = nullptr;  // < 执行图句柄，默认初始化为 nullptr
};

}  // namespace deploy
