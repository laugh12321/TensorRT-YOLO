#include "deploy/core/core.hpp"

#include <iostream>

namespace deploy {

void TRTLogger::log(nvinfer1::ILogger::Severity severity,
                    const char*                 msg) noexcept {
    if (severity > severity_) return;
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

void EngineContext::Destroy() {
    context.reset();
    engine.reset();
    runtime.reset();
}

bool EngineContext::Construct(const void* data, size_t size) {
    Destroy();

    if (data == nullptr || size == 0) return false;

    runtime = std::shared_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(logger_), [](nvinfer1::IRuntime* ptr) {
            if (ptr != nullptr) ptr->destroy();
        });
    if (runtime == nullptr) return false;

    engine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(data, size, nullptr),
        [](nvinfer1::ICudaEngine* ptr) {
            if (ptr != nullptr) ptr->destroy();
        });
    if (engine == nullptr) return false;

    context = std::shared_ptr<nvinfer1::IExecutionContext>(
        engine->createExecutionContext(), [](nvinfer1::IExecutionContext* ptr) {
            if (ptr != nullptr) ptr->destroy();
        });
    return context != nullptr;
}

}  // namespace deploy
