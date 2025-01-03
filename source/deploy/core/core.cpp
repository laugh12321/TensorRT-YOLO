#include <iostream>

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"

namespace deploy {

void TrtLogger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity > mSeverity) return;
    switch (severity) {
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case nvinfer1::ILogger::Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case nvinfer1::ILogger::Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "VERBOSE: ";
            break;
    }
    std::cerr << msg << '\n';
}

void EngineContext::destroy() {
    mContext.reset();
    mEngine.reset();
    mRuntime.reset();
}

bool EngineContext::construct(const void* data, size_t size) {
    destroy();

    if (data == nullptr || size == 0) return false;

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(mLogger), [](nvinfer1::IRuntime* ptr) {
            if (ptr != nullptr) delete ptr;
        });
    if (mRuntime == nullptr) return false;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(data, size),
        [](nvinfer1::ICudaEngine* ptr) {
            if (ptr != nullptr) delete ptr;
        });
    if (mEngine == nullptr) return false;

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(), [](nvinfer1::IExecutionContext* ptr) {
            if (ptr != nullptr) delete ptr;
        });
    return mContext != nullptr;
}

void CudaGraph::destroy() {
    if (graphExec_) {
        cudaGraphExecDestroy(graphExec_);
        graphExec_ = nullptr;  // Clear the pointer to avoid dangling pointer
    }
    if (graph_) {
        cudaGraphDestroy(graph_);
        graph_ = nullptr;  // Clear the pointer
    }
}

void CudaGraph::beginCapture(cudaStream_t stream) {
    CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
}

void CudaGraph::endCapture(cudaStream_t stream) {
    CHECK(cudaStreamEndCapture(stream, &graph_));
    CHECK(cudaGraphInstantiate(&graphExec_, graph_, nullptr, nullptr, 0));
}

void CudaGraph::launch(cudaStream_t stream) {
    CHECK(cudaGraphLaunch(graphExec_, stream));
    CHECK(cudaStreamSynchronize(stream));
}

void CudaGraph::initializeNodes(size_t num) {
    if (num == 0) {
        CHECK(cudaGraphGetNodes(graph_, nullptr, &num));
    }

    if (num > 0) {
        nodes_ = std::make_unique<cudaGraphNode_t[]>(num);
        CHECK(cudaGraphGetNodes(graph_, nodes_.get(), &num));
    } else {
        throw std::runtime_error("Failed to initialize nodes: graph has no nodes.");
    }
}

void CudaGraph::updateKernelNodeParams(size_t index, void** kernelParams) {
    // Check if the node type is kernel
    if (getNodeType(index) != cudaGraphNodeTypeKernel) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a kernel node.");
    }

    // Get current kernel node parameters
    cudaKernelNodeParams kernelNodeParams;
    CHECK(cudaGraphKernelNodeGetParams(nodes_[index], &kernelNodeParams));

    // Modify kernelParams as needed (e.g., changing kernel configuration)
    kernelNodeParams.kernelParams = kernelParams;

    // Update the kernel node with new parameters
    CHECK(cudaGraphExecKernelNodeSetParams(graphExec_, nodes_[index], &kernelNodeParams));
}

void CudaGraph::updateMemcpyNodeParams(size_t index, void* src, void* dst, size_t size) {
    // Check if the node type is memcpy
    if (getNodeType(index) != cudaGraphNodeTypeMemcpy) {
        throw std::runtime_error("Node at index " + std::to_string(index) + " is not a memcpy node.");
    }

    // Get current memcpy node parameters
    cudaMemcpy3DParms memcpyNodeParams;
    CHECK(cudaGraphMemcpyNodeGetParams(nodes_[index], &memcpyNodeParams));

    // Set source and destination pointers
    memcpyNodeParams.srcPtr = make_cudaPitchedPtr(src, size, size, 1);
    memcpyNodeParams.dstPtr = make_cudaPitchedPtr(dst, size, size, 1);
    memcpyNodeParams.extent = make_cudaExtent(size, 1, 1);

    // Update the memcpy node with new parameters
    CHECK(cudaGraphExecMemcpyNodeSetParams(graphExec_, nodes_[index], &memcpyNodeParams));
}

// Helper function to get node type, throwing exception if it's invalid
cudaGraphNodeType CudaGraph::getNodeType(size_t index) {
    cudaGraphNodeType nodeType;
    CHECK(cudaGraphNodeGetType(nodes_[index], &nodeType));
    return nodeType;
}

}  // namespace deploy
