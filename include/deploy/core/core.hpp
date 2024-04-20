#pragma once

#include <NvInferPlugin.h>
#include <NvInferRuntimeBase.h>

#include <memory>

namespace deploy {

/**
 * @brief Custom logger for TensorRT messages.
 */
class TRTLogger : public nvinfer1::ILogger {
private:
    nvinfer1::ILogger::Severity severity_;

public:
    explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO)
        : severity_(severity) {
    }

    void log(nvinfer1::ILogger::Severity severity,
             const char*                 msg) noexcept override;
};

/**
 * @brief Manages TensorRT engine and execution context.
 */
class EngineContext {
private:
    TRTLogger logger_{nvinfer1::ILogger::Severity::kERROR};

    /**
     * @brief Destroys the engine, execution context, and runtime.
     */
    void Destroy();

public:
    std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine  = nullptr;
    std::shared_ptr<nvinfer1::IRuntime>          runtime = nullptr;

    /**
     * @brief Constructs the EngineContext object.
     */
    EngineContext() {
        initLibNvInferPlugins(&logger_, "");
    }

    EngineContext(const EngineContext&)            = default;
    EngineContext(EngineContext&&)                 = delete;
    EngineContext& operator=(const EngineContext&) = default;
    EngineContext& operator=(EngineContext&&)      = delete;
    /**
     * @brief Destroys the EngineContext object and releases associated
     * resources.
     */
    ~EngineContext() {
        Destroy();
    }

    /**
     * @brief Constructs the engine and execution context from serialized data.
     *
     * @param data Pointer to the serialized engine data.
     * @param size Size of the serialized engine data.
     * @return bool True if construction succeeds, false otherwise.
     */
    bool Construct(const void* data, size_t size);
};

}  // namespace deploy
