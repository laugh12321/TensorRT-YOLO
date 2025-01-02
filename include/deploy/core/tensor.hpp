#pragma once

#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "deploy/core/memory.hpp"
#include "deploy/core/types.hpp"

namespace deploy {

/**
 * @brief Struct representing Tensor information.
 */
struct TensorInfo {
private:
    size_t typeSz{};                          /**< Size of the tensor's data type in bytes. */

public:
    std::string                 name{};       /**< Name of the tensor. */
    nvinfer1::Dims              dims{};       /**< Dimensions of the tensor. */
    int64_t                     bytes{};      /**< Total size of the tensor's data in bytes. */
    MemoryManager<PinnedMemory> buffer{};     /**< Buffer object associated with this TensorInfo. */
    bool                        input{false}; /**< Indicates if the tensor is an input tensor. */

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
          buffer(MemoryManager<PinnedMemory>()),
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