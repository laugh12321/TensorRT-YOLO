#pragma once

#include <NvInferPlugin.h>
#include <cuda_runtime.h>

#include <memory>

namespace deploy {

/**
 * @brief Custom logger for handling TensorRT messages.
 */
class TrtLogger : public nvinfer1::ILogger {
private:
    nvinfer1::ILogger::Severity mSeverity; /**< Severity level for logging. */

public:
    /**
     * @brief Constructs a TrtLogger object with the specified severity level.
     *
     * @param severity Severity level for logging (default is INFO).
     */
    explicit TrtLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) : mSeverity(severity) {}

    /**
     * @brief Logs a message with the specified severity level.
     *
     * @param severity Severity level of the message.
     * @param msg Message to be logged.
     */
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override;
};

/**
 * @brief Manages the TensorRT engine and execution context.
 */
class EngineContext {
private:
    TrtLogger mLogger{nvinfer1::ILogger::Severity::kERROR}; /**< Logger for handling TensorRT messages. */

    /**
     * @brief Destroys the engine, execution context, and runtime.
     */
    void destroy();

public:
    std::shared_ptr<nvinfer1::IExecutionContext> mContext = nullptr; /**< Execution context for TensorRT engine. */
    std::shared_ptr<nvinfer1::ICudaEngine>       mEngine  = nullptr; /**< TensorRT engine. */
    std::shared_ptr<nvinfer1::IRuntime>          mRuntime = nullptr; /**< TensorRT runtime. */

    /**
     * @brief Constructs an EngineContext object.
     */
    EngineContext() {
        initLibNvInferPlugins(&mLogger, ""); /**< Initializes TensorRT plugins with custom logger. */
    }

    EngineContext(const EngineContext&)            = default;
    EngineContext(EngineContext&&)                 = delete;
    EngineContext& operator=(const EngineContext&) = default;
    EngineContext& operator=(EngineContext&&)      = delete;
    /**
     * @brief Destroys the EngineContext object and releases associated resources.
     */
    ~EngineContext() {
        destroy(); /**< Destroys the EngineContext object and releases associated resources. */
    }

    /**
     * @brief Constructs the engine and execution context from serialized data.
     *
     * @param data Pointer to the serialized engine data.
     * @param size Size of the serialized engine data.
     * @return bool True if construction succeeds, false otherwise.
     */
    bool construct(const void* data, size_t size);
};

class CudaGraph {
public:
    /**
     * @brief Default constructor
     *
     * Initializes the `CudaGraph` object. The graph and execution graph handles are set to nullptr by default.
     */
    explicit CudaGraph() = default;

    /**
     * @brief Deleted copy constructor
     *
     * Disables the copy constructor to prevent object copying.
     */
    CudaGraph(const CudaGraph&) = delete;

    /**
     * @brief Deleted copy assignment operator
     *
     * Disables the copy assignment operator to prevent object assignment.
     */
    CudaGraph& operator=(const CudaGraph&) = delete;

    /**
     * @brief Deleted move constructor
     *
     * Disables the move constructor to prevent object movement.
     */
    CudaGraph(CudaGraph&&) = delete;

    /**
     * @brief Deleted move assignment operator
     *
     * Disables the move assignment operator to prevent object move assignment.
     */
    CudaGraph& operator=(CudaGraph&&) = delete;

    /**
     * @brief Destructor
     *
     * Destroys the CUDA graph and execution graph resources, releasing related memory.
     */
    ~CudaGraph() { destroy(); }

    /**
     * @brief Destroys CUDA graph and execution graph resources
     *
     * Destroys `graphExec_` and `graph_`, releasing all resources related to CUDA graphs.
     */
    void destroy();

    /**
     * @brief Begins capturing a CUDA graph
     *
     * Starts capturing CUDA operations in the specified stream. Capture mode is `cudaStreamCaptureModeThreadLocal`.
     *
     * @param stream The CUDA stream to capture.
     */
    void beginCapture(cudaStream_t stream);

    /**
     * @brief Ends capturing a CUDA graph
     *
     * Completes the capture and instantiates the graph execution handle.
     *
     * @param stream The CUDA stream to finish capturing.
     */
    void endCapture(cudaStream_t stream);

    /**
     * @brief Launches the CUDA graph
     *
     * Executes the CUDA graph in the specified stream and synchronizes the stream.
     *
     * @param stream The CUDA stream to launch the graph.
     */
    void launch(cudaStream_t stream);

    /**
     * @brief Initializes nodes in the graph
     *
     * Retrieves and initializes nodes in the CUDA graph. If `num` is 0, the node count is automatically fetched.
     *
     * @param num The number of nodes to initialize.
     * @throws std::runtime_error If the graph contains no nodes.
     */
    void initializeNodes(size_t num = 0);

    /**
     * @brief Updates kernel node parameters
     *
     * Updates the parameters of the specified kernel node.
     *
     * @param index The index of the node.
     * @param kernelParams The new kernel parameters.
     *
     * @throws std::runtime_error If the specified node is not a kernel node.
     */
    void updateKernelNodeParams(size_t index, void** kernelParams);

    /**
     * @brief Updates memcpy node parameters
     *
     * Updates the parameters of the specified memcpy node.
     *
     * @param index The index of the node.
     * @param src The source memory address.
     * @param dst The destination memory address.
     * @param size The size to copy.
     *
     * @throws std::runtime_error If the specified node is not a memcpy node.
     */
    void updateMemcpyNodeParams(size_t index, void* src, void* dst, size_t size);

private:
    /**
     * @brief Retrieves the node type
     *
     * Retrieves the type of the node at the specified index.
     * Throws an exception if the node type is invalid.
     *
     * @param index The node index.
     * @return cudaGraphNodeType The type of the node.
     */
    cudaGraphNodeType getNodeType(size_t index);

private:
    std::unique_ptr<cudaGraphNode_t[]> nodes_;                ///< Unique pointer to store graph nodes
    cudaGraph_t                        graph_     = nullptr;  ///< Graph handle, initialized to nullptr
    cudaGraphExec_t                    graphExec_ = nullptr;  ///< Execution graph handle, initialized to nullptr
};

}  // namespace deploy
