#include <iostream>

#include "deploy/core/core.hpp"

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
            if (ptr != nullptr) ptr->destroy();
        });
    if (mRuntime == nullptr) return false;

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(data, size, nullptr),
        [](nvinfer1::ICudaEngine* ptr) {
            if (ptr != nullptr) ptr->destroy();
        });
    if (mEngine == nullptr) return false;

    mContext = std::shared_ptr<nvinfer1::IExecutionContext>(
        mEngine->createExecutionContext(), [](nvinfer1::IExecutionContext* ptr) {
            if (ptr != nullptr) ptr->destroy();
        });
    return mContext != nullptr;
}

}  // namespace deploy
