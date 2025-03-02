/**
 * @file core.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 实现了 TensorRT 日志记录器和 CUDA 图管理类的功能
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"

namespace deploy {

// 定义日志级别与前缀的映射表
const std::map<nvinfer1::ILogger::Severity, std::string> TRTLogger::severity_map_ = {
    {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
    {         nvinfer1::ILogger::Severity::kERROR,          "ERROR: "},
    {       nvinfer1::ILogger::Severity::kWARNING,        "WARNING: "},
    {          nvinfer1::ILogger::Severity::kINFO,           "INFO: "},
    {       nvinfer1::ILogger::Severity::kVERBOSE,        "VERBOSE: "}
};

// 构造函数
TRTLogger::TRTLogger(nvinfer1::ILogger::Severity severity) : severity_(severity) {}

// 实现 log 方法
void TRTLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    // 如果当前日志级别高于设置的日志级别，则忽略该日志
    if (severity > severity_) return;

    // 根据日志级别选择输出流（INFO 及以上输出到标准输出，其他输出到标准错误）
    std::ostream& stream = severity >= nvinfer1::ILogger::Severity::kINFO ? std::cout : std::cerr;

    // 查找日志级别的映射表，获取对应的前缀
    auto it = severity_map_.find(severity);
    if (it != severity_map_.end()) {
        stream << it->second << msg << '\n';  // 输出日志
    }
}

// 默认构造函数
TRTManager::TRTManager() : context_(nullptr), engine_(nullptr), runtime_(nullptr), logger_(nullptr) {}

// 初始化方法
void TRTManager::initialize(void const* blob, std::size_t size) {
    logger_ = std::make_unique<TRTLogger>(nvinfer1::ILogger::Severity::kWARNING);

    initLibNvInferPlugins(logger_.get(), "");

    // 创建 TensorRT runtime
    runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) {
        throw std::runtime_error("Failed to create TensorRT runtime.");
    }

    // 反序列化引擎
    engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(blob, size));
    if (!engine_) {
        throw std::runtime_error("Failed to deserialize CUDA engine.");
    }

    // 创建执行上下文
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context.");
    }
}

// 克隆方法
std::unique_ptr<TRTManager> TRTManager::clone() const {
    if (!engine_ || !runtime_) {
        throw std::runtime_error("Invalid engine or runtime in TRTManager.");
    }

    // 创建新的 TRTManager 实例
    auto newManager = std::make_unique<TRTManager>();

    // 共享 engine
    newManager->engine_ = engine_;

    // 创建新的 context
    newManager->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(newManager->engine_->createExecutionContext());
    if (!newManager->context_) {
        throw std::runtime_error("Failed to create new execution context during clone.");
    }

    return newManager;
}

// 析构函数
TRTManager::~TRTManager() {
    context_.reset();
    engine_.reset();
    runtime_.reset();
    logger_.reset();
}

// nvinfer1::IExecutionContext 相关方法
bool TRTManager::setTensorAddress(char const* tensorName, void* data) {
    return context_->setTensorAddress(tensorName, data);
}

bool TRTManager::setInputShape(char const* tensorName, nvinfer1::Dims const& dims) {
    return context_->setInputShape(tensorName, dims);
}

bool TRTManager::enqueueV3(cudaStream_t stream) {
    return context_->enqueueV3(stream);
}

// nvinfer1::ICudaEngine 相关方法
nvinfer1::Dims TRTManager::getTensorShape(char const* tensorName) const noexcept {
    return engine_->getTensorShape(tensorName);
}

nvinfer1::DataType TRTManager::getTensorDataType(char const* tensorName) const noexcept {
    return engine_->getTensorDataType(tensorName);
}

nvinfer1::TensorIOMode TRTManager::getTensorIOMode(char const* tensorName) const noexcept {
    return engine_->getTensorIOMode(tensorName);
}

nvinfer1::Dims TRTManager::getProfileShape(char const* tensorName, int32_t profileIndex, nvinfer1::OptProfileSelector select) const noexcept {
    return engine_->getProfileShape(tensorName, profileIndex, select);
}

int32_t TRTManager::getNbIOTensors() const noexcept {
    return engine_->getNbIOTensors();
}

char const* TRTManager::getIOTensorName(int32_t index) const noexcept {
    return engine_->getIOTensorName(index);
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
