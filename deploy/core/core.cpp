/**
 * @file core.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 实现了 TensorRT 日志记录器和 CUDA 图管理类的功能
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <map>

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"

namespace deploy {

// 定义自定义日志记录器 TRTLogger
struct TRTManager::TRTLogger : public nvinfer1::ILogger {
    TRTManager& parent;  // 指向 TRTManager 的引用，用于访问父类的成员

    // 构造函数，初始化父类引用
    TRTLogger(TRTManager& parent) : parent(parent) {}

    // 重写 TensorRT 的日志记录方法
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
        // 如果日志级别低于 WARNING，则忽略该日志
        if (severity > nvinfer1::ILogger::Severity::kWARNING) return;

        // 根据日志级别选择输出流（INFO 及以上输出到标准输出，其他输出到标准错误）
        std::ostream& stream = severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;

        // 查找日志级别的映射表，获取对应的前缀
        auto it = severity_map_.find(severity);
        if (it != severity_map_.end()) {
            stream << it->second << msg << '\n';  // 输出日志
        }
    }

private:
    // 日志级别与前缀的映射表
    std::map<nvinfer1::ILogger::Severity, std::string> severity_map_ = {
        {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
        {         nvinfer1::ILogger::Severity::kERROR,          "ERROR: "},
        {       nvinfer1::ILogger::Severity::kWARNING,        "WARNING: "},
        {          nvinfer1::ILogger::Severity::kINFO,           "INFO: "},
        {       nvinfer1::ILogger::Severity::kVERBOSE,        "VERBOSE: "}
    };
};

// 初始化静态成员变量：引用计数器
std::atomic<int> TRTManager::refCount(0);

TRTManager& TRTManager::instance() {
    static TRTManager manager;  // < 静态局部变量，线程安全
    return manager;
}

nvinfer1::IRuntime* TRTManager::getRuntime() {
    refCount.fetch_add(1, std::memory_order_acq_rel);  // < 增加引用计数
    return instance().runtime.get();                   // < 返回运行时对象
}

void TRTManager::engineDeleter(nvinfer1::ICudaEngine* engine) {
    if (engine) {
        delete engine;  // < 调用 TensorRT 的销毁方法释放引擎
    }
    // 减少引用计数，当引用计数为 0 时释放运行时对象
    if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        instance().runtime.reset();
    }
}

TRTManager::TRTManager() {
    // 创建自定义日志记录器
    logger = std::make_unique<TRTLogger>(*this);

    // 初始化 TensorRT 插件库
    initLibNvInferPlugins(logger.get(), "");

    // 创建 TensorRT 运行时对象
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger));
    if (!runtime) throw std::runtime_error("Failed to call createInferRuntime().");
}

void CudaGraph::destroy() {
    // 销毁执行图资源
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;  // 清空指针以避免悬空指针
    }
    // 销毁图资源
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;  // 清空指针
    }
}

void CudaGraph::beginCapture(cudaStream_t stream) {
    // 开始捕获 CUDA 图操作，捕获模式为 `cudaStreamCaptureModeThreadLocal`
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

void CudaGraph::endCapture(cudaStream_t stream) {
    // 结束捕获 CUDA 图操作，获取图句柄
    CHECK(cudaStreamEndCapture(stream, &graph_));
    // 实例化图执行句柄
    CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
}

void CudaGraph::launch(cudaStream_t stream) {
    // 启动执行图，并同步流
    CHECK(cudaGraphLaunch(graphExec_, stream));
    CHECK(cudaStreamSynchronize(stream));
}

void CudaGraph::initializeNodes(size_t num) {
    // 如果节点数量为0，自动获取图中的节点数量
    if (num == 0) {
        CHECK(cudaGraphGetNodes(graph_, nullptr, &num));
    }

    if (num > 0) {
        // 为节点分配内存
        nodes_ = std::make_unique<cudaGraphNode_t[]>(num);
        CHECK(cudaGraphGetNodes(graph_, nodes_.get(), &num));
    } else {
        // 如果图中没有节点，抛出异常
        throw std::runtime_error("Failed to initialize nodes: graph has no nodes.");
    }
}

void CudaGraph::updateKernelNodeParams(size_t index, void** kernelParams) {
    // 检查指定节点是否为内核节点
    if (getNodeType(index) != cudaGraphNodeTypeKernel) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a kernel node.");
    }

    // 获取当前内核节点的参数
    cudaKernelNodeParams kernelNodeParams;
    CHECK(cudaGraphKernelNodeGetParams(nodes_[index], &kernelNodeParams));

    // 根据需要修改内核参数（例如，更改内核配置）
    kernelNodeParams.kernelParams = kernelParams;

    // 更新内核节点的参数
    CHECK(cudaGraphExecKernelNodeSetParams(graphExec_, nodes_[index], &kernelNodeParams));
}

void CudaGraph::updateMemcpyNodeParams(size_t index, void* src, void* dst, size_t size) {
    // 检查指定节点是否为 memcpy 节点
    if (getNodeType(index) != cudaGraphNodeTypeMemcpy) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a memcpy node.");
    }

    // 获取当前 memcpy 节点的参数
    cudaMemcpy3DParms memcpyNodeParams;
    CHECK(cudaGraphMemcpyNodeGetParams(nodes_[index], &memcpyNodeParams));

    // 设置源和目标内存地址
    memcpyNodeParams.srcPtr = make_cudaPitchedPtr(src, size, size, 1);
    memcpyNodeParams.dstPtr = make_cudaPitchedPtr(dst, size, size, 1);
    memcpyNodeParams.extent = make_cudaExtent(size, 1, 1);

    // 更新 memcpy 节点的参数
    CHECK(cudaGraphExecMemcpyNodeSetParams(graphExec_, nodes_[index], &memcpyNodeParams));
}

cudaGraphNodeType CudaGraph::getNodeType(size_t index) {
    cudaGraphNodeType nodeType;
    CHECK(cudaGraphNodeGetType(nodes_[index], &nodeType));
    return nodeType;
}

}  // namespace deploy
