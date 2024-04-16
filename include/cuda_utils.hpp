#pragma once

#include <NvInferPlugin.h>
#include <Nvinfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace cuda_utils {

#define CUDA_CHECK_ERROR(code) __cuda_check_error((code), __FILE__, __LINE__)

static inline bool __cuda_check_error(cudaError_t code, const char *file,
                                      int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error:\n";
        std::cerr << "    File:       " << file << "\n";
        std::cerr << "    Line:       " << line << "\n";
        std::cerr << "    Error code: " << code << "\n";
        std::cerr << "    Error text: " << cudaGetErrorString(code) << "\n";
        return false;
    }
    return true;
}

class Logger : public nvinfer1::ILogger {
   public:
    nvinfer1::ILogger::Severity reportableSeverity;

    explicit Logger(nvinfer1::ILogger::Severity severity =
                        nvinfer1::ILogger::Severity::kINFO)
        : reportableSeverity(severity) {}

    void log(nvinfer1::ILogger::Severity severity,
             const char                 *msg) noexcept override {
        if (severity > reportableSeverity) return;

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
        std::cerr << msg << std::endl;
    }
};  // class Logger

inline size_t dataTypeSize(nvinfer1::DataType dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4U;
        case nvinfer1::DataType::kHALF:
            return 2U;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8:
            return 1U;
    }
    return 0;
}

inline int64_t volume(nvinfer1::Dims const &dims) {
    return std::accumulate(dims.d, dims.d + dims.nbDims, int64_t{1},
                           std::multiplies<int64_t>{});
}

inline int upbound(int n, int align = 32) {
    return (n + align - 1) / align * align;
}

class HostDeviceMem {
   private:
    void   *_host            = nullptr;
    void   *_device          = nullptr;
    size_t  _dtype_bytes     = 0;
    int64_t _host_bytes      = 0;
    int64_t _device_bytes    = 0;
    int64_t _host_capacity   = 0;
    int64_t _device_capacity = 0;

    void reallocHost(int64_t bytes) {
        if (_host_capacity < bytes) {
            cudaFreeHost(_host);
            cudaMallocHost(&_host, bytes);
            _host_capacity = bytes;
        }
        _host_bytes = bytes;
    }

    void reallocDevice(int64_t bytes) {
        if (_device_capacity < bytes) {
            cudaFree(_device);
            cudaMalloc(&_device, bytes);
            _device_capacity = bytes;
        }
        _device_bytes = bytes;
    }

   public:
    HostDeviceMem(size_t dtype_bytes) : _dtype_bytes(dtype_bytes) {}

    // HostDeviceMem(const HostDeviceMem &other)            = delete;
    HostDeviceMem &operator=(const HostDeviceMem &other) = delete;

    ~HostDeviceMem() {
        cudaFreeHost(_host);
        cudaFree(_device);
    }

    void *host() const { return _host; }

    void *host(int64_t size) {
        reallocHost(size * _dtype_bytes);
        return _host;
    }

    void *device() const { return _device; }

    void *device(int64_t size) {
        reallocDevice(size * _dtype_bytes);
        return _device;
    }

    int64_t hostSize() const { return _host_bytes / _dtype_bytes; }

    int64_t deviceSize() const { return _device_bytes / _dtype_bytes; }
};  // class HostDeviceMem

class TimerBase {
   public:
    virtual void start() {}
    virtual void stop() {}
    float        microseconds() const noexcept { return mMs * 1000.F; }
    float        milliseconds() const noexcept { return mMs; }
    float        seconds() const noexcept { return mMs / 1000.F; }
    void         reset() noexcept { mMs = 0.F; }

   protected:
    float mMs{0.0F};
};

class GpuTimer : public TimerBase {
   public:
    explicit GpuTimer(cudaStream_t stream = 0) : mStream(stream) {
        CUDA_CHECK_ERROR(cudaEventCreate(&mStart));
        CUDA_CHECK_ERROR(cudaEventCreate(&mStop));
    }
    ~GpuTimer() {
        CUDA_CHECK_ERROR(cudaEventDestroy(mStart));
        CUDA_CHECK_ERROR(cudaEventDestroy(mStop));
    }
    void start() override {
        CUDA_CHECK_ERROR(cudaEventRecord(mStart, mStream));
    }
    void stop() override {
        CUDA_CHECK_ERROR(cudaEventRecord(mStop, mStream));
        float ms{0.0F};
        CUDA_CHECK_ERROR(cudaEventSynchronize(mStop));
        CUDA_CHECK_ERROR(cudaEventElapsedTime(&ms, mStart, mStop));
        mMs += ms;
    }

   private:
    cudaEvent_t  mStart, mStop;
    cudaStream_t mStream;
};  // class GpuTimer

template <typename Clock>
class CpuTimer : public TimerBase {
   public:
    using clock_type = Clock;

    void start() override { mStart = Clock::now(); }
    void stop() override {
        mStop = Clock::now();
        mMs += std::chrono::duration<float, std::milli>{mStop - mStart}.count();
    }

   private:
    std::chrono::time_point<Clock> mStart, mStop;
};  // class CpuTimer

template <typename T>
static void destroy_ptr(T *ptr) {
    if (ptr) ptr->destroy();
}

class EngineContext {
   public:
    EngineContext() { initLibNvInferPlugins(&gLogger, ""); }

    std::shared_ptr<nvinfer1::IExecutionContext> context = nullptr;
    std::shared_ptr<nvinfer1::ICudaEngine>       engine  = nullptr;
    std::shared_ptr<nvinfer1::IRuntime>          runtime = nullptr;

    virtual ~EngineContext() { destroy(); }

    bool construct(const void *pdata, size_t size) {
        destroy();

        if (pdata == nullptr || size == 0) return false;

        runtime = std::shared_ptr<nvinfer1::IRuntime>(
            nvinfer1::createInferRuntime(gLogger),
            destroy_ptr<nvinfer1::IRuntime>);
        if (runtime == nullptr) return false;

        engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(pdata, size, nullptr),
            destroy_ptr<nvinfer1::ICudaEngine>);
        if (engine == nullptr) return false;

        context = std::shared_ptr<nvinfer1::IExecutionContext>(
            engine->createExecutionContext(),
            destroy_ptr<nvinfer1::IExecutionContext>);
        return context != nullptr;
    }

   private:
    void destroy() {
        context.reset();
        engine.reset();
        runtime.reset();
    }

    Logger gLogger{nvinfer1::ILogger::Severity::kERROR};
};

}  // namespace cuda_utils
