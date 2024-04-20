
#include "deploy/core/tensor.hpp"

#include "deploy/core/macro.hpp"

namespace deploy {

void Tensor::ReallocHost(int64_t bytes) {
    if (host_capacity_ < bytes) {
        CUDA_CHECK_ERROR(cudaFreeHost(host_ptr_));
        CUDA_CHECK_ERROR(cudaMallocHost(&host_ptr_, bytes));
        host_capacity_ = bytes;
    }
    host_bytes_ = bytes;
}

void Tensor::ReallocDevice(int64_t bytes) {
    if (device_capacity_ < bytes) {
        CUDA_CHECK_ERROR(cudaFree(device_ptr_));
        CUDA_CHECK_ERROR(cudaMalloc(&device_ptr_, bytes));
        device_capacity_ = bytes;
    }
    device_bytes_ = bytes;
}

Tensor::~Tensor() {
    if (host_ptr_ != nullptr) {
        CUDA_CHECK_ERROR(cudaFreeHost(host_ptr_));
    }
    if (device_ptr_ != nullptr) {
        CUDA_CHECK_ERROR(cudaFree(device_ptr_));
    }
}

void* Tensor::Host(int64_t size) {
    ReallocHost(size * dtype_bytes_);
    return host_ptr_;
}

void* Tensor::Device(int64_t size) {
    ReallocDevice(size * dtype_bytes_);
    return device_ptr_;
}

}  // namespace deploy