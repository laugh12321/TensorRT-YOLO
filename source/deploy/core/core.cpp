/**
 * @file core.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 实现了 TensorRT 日志记录器和 CUDA 图管理类的功能
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <cstddef>
#include <cstring>
#include <stdexcept>

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"

namespace deploy {

std::map<nvinfer1::ILogger::Severity, std::string> TrtLogger::severity_map_ = {
    {nvinfer1::ILogger::Severity::kINTERNAL_ERROR, "INTERNAL_ERROR: "},
    {         nvinfer1::ILogger::Severity::kERROR,          "ERROR: "},
    {       nvinfer1::ILogger::Severity::kWARNING,        "WARNING: "},
    {          nvinfer1::ILogger::Severity::kINFO,           "INFO: "},
    {       nvinfer1::ILogger::Severity::kVERBOSE,        "VERBOSE: "}
};

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
