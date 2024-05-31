#pragma once

#include <NvInferRuntimeBase.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "deploy/core/types.hpp"

namespace deploy {

/**
 * @brief Class representing a Tensor.
 */
class Tensor {
private:
    void*   hostPtr     = nullptr; /**< Pointer to host memory */
    void*   devicePtr   = nullptr; /**< Pointer to device memory */
    int64_t hostBytes   = 0;       /**< Size of host memory in bytes */
    int64_t deviceBytes = 0;       /**< Size of device memory in bytes */
    int64_t hostCap     = 0;       /**< Capacity of host memory in bytes */
    int64_t deviceCap   = 0;       /**< Capacity of device memory in bytes */

    /**
     * @brief Reallocates host memory to the specified size.
     *
     * @param bytes Size of host memory to allocate in bytes.
     */
    void reallocHost(int64_t bytes);

    /**
     * @brief Reallocates device memory to the specified size.
     *
     * @param bytes Size of device memory to allocate in bytes.
     */
    void reallocDevice(int64_t bytes);

public:
    explicit Tensor() {}

    ~Tensor(); /**< Destructor */

    /**
     * @brief Accessor for the pointer to host memory.
     *
     * @return void* Pointer to host memory.
     */
    void* host() {
        return hostPtr;
    }

    /**
     * @brief Allocates host memory to the specified size.
     *
     * @param bytes Size of host memory to allocate in bytes.
     * @return void* Pointer to allocated host memory.
     */
    void* host(int64_t bytes);

    /**
     * @brief Accessor for the pointer to device memory.
     *
     * @return void* Pointer to device memory.
     */
    void* device() {
        return devicePtr;
    }

    /**
     * @brief Allocates device memory to the specified size.
     *
     * @param bytes Size of device memory to allocate in bytes.
     * @return void* Pointer to allocated device memory.
     */
    void* device(int64_t bytes);
};

/**
 * @brief Struct representing Tensor information.
 */
struct TensorInfo {
private:
    size_t typeSz{};             /**< Size of the tensor's data type in bytes. */

public:
    std::string    name{};       /**< Name of the tensor. */
    nvinfer1::Dims dims{};       /**< Dimensions of the tensor. */
    int64_t        bytes{};      /**< Total size of the tensor's data in bytes. */
    Tensor         tensor{};     /**< Tensor object associated with this TensorInfo. */
    bool           input{false}; /**< Indicates if the tensor is an input tensor. */

    /**
     * @brief Constructs a TensorInfo object with the given parameters.
     *
     * @param name Name of the tensor.
     * @param dims Dimensions of the tensor.
     * @param input Indicates if the tensor is an input tensor.
     * @param typeSz Size of the tensor's data type in bytes.
     * @param bytes Total size of the tensor's data in bytes.
     */
    TensorInfo(const char* name, const nvinfer1::Dims& dims, bool input, size_t typeSz, int64_t bytes)
        : name(name),
          dims(dims),
          input(input),
          typeSz(typeSz),
          tensor(Tensor()),
          bytes(bytes) {
    }

    /**
     * @brief Updates the total size of the tensor's data based on its dimensions and data type size.
     */
    void update() {
        bytes = calculateVolume(dims) * typeSz;
    }
};

}  // namespace deploy