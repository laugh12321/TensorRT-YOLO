
#include "deploy/core/macro.hpp"
#include "deploy/core/tensor.hpp"

namespace deploy {

void Tensor::reallocHost(int64_t bytes) {
    if (hostCap < bytes) {
        CHECK(cudaFreeHost(hostPtr));
        CHECK(cudaMallocHost(&hostPtr, bytes));
        hostCap = bytes;
    }
    hostBytes = bytes;
}

void Tensor::reallocDevice(int64_t bytes) {
    if (deviceCap < bytes) {
        CHECK(cudaFree(devicePtr));
        CHECK(cudaMalloc(&devicePtr, bytes));
        deviceCap = bytes;
    }
    deviceBytes = bytes;
}

Tensor::~Tensor() {
    if (hostPtr != nullptr) {
        CHECK(cudaFreeHost(hostPtr));
    }
    if (devicePtr != nullptr) {
        CHECK(cudaFree(devicePtr));
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