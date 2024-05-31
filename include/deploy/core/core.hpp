#pragma once

#include <NvInferPlugin.h>
#include <NvInferRuntimeBase.h>

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

}  // namespace deploy
