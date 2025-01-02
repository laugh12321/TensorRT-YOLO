#pragma once

#include <cuda_runtime.h>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @class MemoryManager
 * @brief CUDA memory manager class providing memory allocation, deallocation, and memory transfer between host and device.
 *
 * This is a template class that allows managing memory resources with different types (e.g., Pinned Memory, Unified Memory, Zero-Copy Memory).
 * It provides interfaces for allocating memory, deallocating memory, and transferring data between host and device.
 * Additionally, copy constructor and copy assignment operator are deleted, while move constructor and move assignment operator are supported.
 *
 * @tparam MemType Memory type, which must define static methods like `allocate`, `free`, `hostToDevice`, and `deviceToHost`.
 */
template <typename MemType>
class MemoryManager {
public:
    /**
     * @brief Constructor to initialize the memory manager.
     *
     * Initializes a new `MemoryManager` instance with size 0, and both host and device pointers set to nullptr.
     */
    MemoryManager() : size_(0), host_(nullptr), device_(nullptr) {}

    /**
     * @brief Deleted copy constructor.
     *
     * Disables the default copy constructor to avoid resource management issues during object copy.
     */
    MemoryManager(const MemoryManager&) = delete;

    /**
     * @brief Deleted copy assignment operator.
     *
     * Disables the default copy assignment operator to avoid resource management issues during assignment.
     */
    MemoryManager& operator=(const MemoryManager&) = delete;

    /**
     * @brief Move constructor.
     *
     * Moves resources from another `MemoryManager` object to the current one. The original object will have its resources reset.
     *
     * @param other The source object from which resources are moved.
     */
    MemoryManager(MemoryManager&& other) noexcept;

    /**
     * @brief Move assignment operator.
     *
     * Moves resources from another `MemoryManager` object to the current one. The original object will have its resources reset.
     *
     * @param other The source object from which resources are moved.
     * @return Returns a reference to the current object.
     */
    MemoryManager& operator=(MemoryManager&& other) noexcept;

    /**
     * @brief Frees allocated memory.
     *
     * Frees both host and device memory, ensuring that the memory is no longer in use.
     */
    void free();

    /**
     * @brief Allocates memory of the specified size.
     *
     * Allocates memory on both host and device for the given size, and updates the memory size.
     *
     * @param size The size of memory to allocate.
     */
    void allocate(size_t size);

    /**
     * @brief Returns the host memory pointer.
     *
     * @return Host memory pointer.
     */
    void* host() { return host_; }

    /**
     * @brief Returns the device memory pointer.
     *
     * @return Device memory pointer.
     */
    void* device() { return device_; }

    /**
     * @brief Returns the size of the current allocated memory.
     *
     * @return The size of the allocated memory.
     */
    size_t size() const { return size_; }

    /**
     * @brief Copies memory from host to device.
     *
     * Asynchronously copies memory from host to device using the provided CUDA stream.
     *
     * @param stream CUDA stream.
     */
    void hostToDevice(cudaStream_t stream) { MemType::hostToDevice(host_, device_, size_, stream); }

    /**
     * @brief Copies memory from device to host.
     *
     * Asynchronously copies memory from device to host using the provided CUDA stream.
     *
     * @param stream CUDA stream.
     */
    void deviceToHost(cudaStream_t stream) { MemType::deviceToHost(host_, device_, size_, stream); }

private:
    void*  host_   = nullptr;  ///< Host memory pointer
    void*  device_ = nullptr;  ///< Device memory pointer
    size_t size_   = 0;        ///< The size of allocated memory
};

/**
 * @brief Pinned Memory type.
 *
 * Pinned memory is memory allocated using `cudaMallocHost` that can be directly accessed.
 */
struct PinnedMemory {
    /**
     * @brief Allocates pinned memory.
     *
     * Allocates pinned memory on both host and device for the specified size.
     *
     * @param host Host memory pointer.
     * @param device Device memory pointer.
     * @param size Size of memory to allocate.
     */
    static void allocate(void*& host, void*& device, size_t size) {
        CHECK(cudaMallocHost(&host, size));
        CHECK(cudaMalloc(&device, size));
    }

    /**
     * @brief Frees pinned memory.
     *
     * Frees pinned memory on both host and device.
     *
     * @param host Host memory pointer.
     * @param device Device memory pointer.
     */
    static void free(void* host, void* device) {
        if (host) CHECK(cudaFreeHost(host));
        if (device) CHECK(cudaFree(device));
    }

    /**
     * @brief Copies memory from host to device.
     *
     * Asynchronously copies memory from host to device.
     *
     * @param host Host memory pointer.
     * @param device Device memory pointer.
     * @param size Size of memory to copy.
     * @param stream CUDA stream.
     */
    static void hostToDevice(void* host, void* device, size_t size, cudaStream_t stream) {
        CHECK(cudaMemcpyAsync(device, host, size, cudaMemcpyHostToDevice, stream));
    }

    /**
     * @brief Copies memory from device to host.
     *
     * Asynchronously copies memory from device to host.
     *
     * @param host Host memory pointer.
     * @param device Device memory pointer.
     * @param size Size of memory to copy.
     * @param stream CUDA stream.
     */
    static void deviceToHost(void* host, void* device, size_t size, cudaStream_t stream) {
        CHECK(cudaMemcpyAsync(host, device, size, cudaMemcpyDeviceToHost, stream));
    }
};

}  // namespace deploy