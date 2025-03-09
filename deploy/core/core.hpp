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

#include <map>
#include <memory>
#include <string>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @class TRTLogger
 * @brief TensorRT 日志记录器类，继承自 nvinfer1::ILogger。
 * 用于管理 TensorRT 的日志记录行为，支持设置日志级别并重写日志记录方法。
 */
class TRTLogger : public nvinfer1::ILogger {
public:
    /**
     * @brief 构造函数，允许设置日志级别，默认为 INFO。
     * @param severity 日志级别，默认为 nvinfer1::ILogger::Severity::kINFO。
     */
    explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO);

    // 禁用拷贝和移动语义
    TRTLogger(const TRTLogger&)            = delete;  // < 禁用拷贝构造函数
    TRTLogger& operator=(const TRTLogger&) = delete;  // < 禁用拷贝赋值运算符
    TRTLogger(TRTLogger&&)                 = delete;  // < 禁用移动构造函数
    TRTLogger& operator=(TRTLogger&&)      = delete;  // < 禁用移动赋值运算符

    /**
     * @brief 重写 TensorRT 的日志记录方法。
     * @param severity 日志级别。
     * @param msg 日志消息。
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;

private:
    nvinfer1::ILogger::Severity severity_;  // < 当前日志级别

    /**
     * @brief 日志级别与前缀的映射表。
     * 用于将日志级别转换为可读的字符串前缀。
     */
    static const std::map<nvinfer1::ILogger::Severity, std::string> severity_map_;
};

/**
 * @class TRTManager
 * @brief TensorRT 管理器类，用于管理 TensorRT 的上下文、引擎和运行时。
 * 提供初始化、克隆、设置张量地址、设置输入形状、执行推理等方法。
 */
class DEPLOYAPI TRTManager {
public:
    /**
     * @brief 默认构造函数。
     */
    TRTManager();

    /**
     * @brief 析构函数。
     */
    ~TRTManager();

    // 禁用拷贝和移动语义
    TRTManager(const TRTManager&)            = delete;  // < 禁用拷贝构造函数
    TRTManager& operator=(const TRTManager&) = delete;  // < 禁用拷贝赋值运算符
    TRTManager(TRTManager&&)                 = delete;  // < 禁用移动构造函数
    TRTManager& operator=(TRTManager&&)      = delete;  // < 禁用移动赋值运算符

    /**
     * @brief 初始化方法，用于加载 TensorRT 引擎。
     * @param blob 包含引擎数据的指针。
     * @param size 引擎数据的大小。
     */
    void initialize(void const* blob, std::size_t size);

    /**
     * @brief 克隆方法，返回一个 TRTManager 的独占指针。
     * @return std::unique_ptr<TRTManager> 克隆的 TRTManager 对象。
     */
    std::unique_ptr<TRTManager> clone() const;

    /**
     * @brief 设置张量地址。
     * @param tensorName 张量名称。
     * @param data 张量数据指针。
     * @return bool 设置是否成功。
     */
    bool setTensorAddress(char const* tensorName, void* data);

    /**
     * @brief 设置输入形状。
     * @param tensorName 张量名称。
     * @param dims 张量维度。
     * @return bool 设置是否成功。
     */
    bool setInputShape(char const* tensorName, nvinfer1::Dims const& dims);

    /**
     * @brief 在指定的 CUDA 流上执行推理。
     * @param stream CUDA 流。
     * @return bool 推理是否成功。
     */
    bool enqueueV3(cudaStream_t stream);

    /**
     * @brief 获取张量的形状。
     * @param tensorName 张量名称。
     * @return nvinfer1::Dims 张量的形状。
     */
    nvinfer1::Dims getTensorShape(char const* tensorName) const noexcept;

    /**
     * @brief 获取张量的数据类型。
     * @param tensorName 张量名称。
     * @return nvinfer1::DataType 张量的数据类型。
     */
    nvinfer1::DataType getTensorDataType(char const* tensorName) const noexcept;

    /**
     * @brief 获取张量的输入输出模式。
     * @param tensorName 张量名称。
     * @return nvinfer1::TensorIOMode 张量的输入输出模式。
     */
    nvinfer1::TensorIOMode getTensorIOMode(char const* tensorName) const noexcept;

    /**
     * @brief 获取指定配置文件的张量形状。
     * @param tensorName 张量名称。
     * @param profileIndex 配置文件索引。
     * @param select 配置文件选择器。
     * @return nvinfer1::Dims 张量的形状。
     */
    nvinfer1::Dims getProfileShape(char const* tensorName, int32_t profileIndex, nvinfer1::OptProfileSelector select) const noexcept;

    /**
     * @brief 获取输入输出张量的数量。
     * @return int32_t 输入输出张量的数量。
     */
    int32_t getNbIOTensors() const noexcept;

    /**
     * @brief 获取指定索引的输入输出张量名称。
     * @param index 张量索引。
     * @return char const* 张量名称。
     */
    char const* getIOTensorName(int32_t index) const noexcept;

private:
    std::unique_ptr<nvinfer1::IExecutionContext> context_;  // < TensorRT 执行上下文
    std::shared_ptr<nvinfer1::ICudaEngine>       engine_;   // < TensorRT CUDA 引擎
    std::unique_ptr<nvinfer1::IRuntime>          runtime_;  // < TensorRT 运行时
    std::unique_ptr<TRTLogger>                   logger_;   // < TensorRT 日志记录器
};

/**
 * @brief CUDA 图类，负责管理 CUDA 图的创建、执行、捕获等操作
 *
 * 该类封装了 CUDA 图的相关操作，包括捕获、执行、初始化节点、更新节点参数等。
 * CUDA 图是一种高效的 CUDA 操作执行模型，可以显著提高一些重复性操作的性能。
 */
class DEPLOYAPI CudaGraph {
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
