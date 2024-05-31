
#include "deploy/core/macro.hpp"
#include "deploy/core/tensor.hpp"

namespace deploy {

void Tensor::reallocHost(int64_t bytes) {
    if (hostCap < bytes) {
        CUDA(cudaFreeHost(hostPtr));
        CUDA(cudaMallocHost(&hostPtr, bytes));
        hostCap = bytes;
    }
    hostBytes = bytes;
}

void Tensor::reallocDevice(int64_t bytes) {
    if (deviceCap < bytes) {
        CUDA(cudaFree(devicePtr));
        CUDA(cudaMalloc(&devicePtr, bytes));
        deviceCap = bytes;
    }
    deviceBytes = bytes;
}

Tensor::~Tensor() {
    if (hostPtr != nullptr) {
        CUDA(cudaFreeHost(hostPtr));
    }
    if (devicePtr != nullptr) {
        CUDA(cudaFree(devicePtr));
    }
}

void* Tensor::host(int64_t size) {
    reallocHost(size);
    return hostPtr;
}

void* Tensor::device(int64_t size) {
    reallocDevice(size);
    return devicePtr;
}

}  // namespace deploy