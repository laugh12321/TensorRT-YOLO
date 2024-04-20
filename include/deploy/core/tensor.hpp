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
    void*   host_ptr_        = nullptr; /**< Pointer to host memory */
    void*   device_ptr_      = nullptr; /**< Pointer to device memory */
    size_t  dtype_bytes_     = 0;       /**< Size of the data type in bytes */
    int64_t host_bytes_      = 0;       /**< Size of host memory in bytes */
    int64_t device_bytes_    = 0;       /**< Size of device memory in bytes */
    int64_t host_capacity_   = 0;       /**< Capacity of host memory in bytes */
    int64_t device_capacity_ = 0;       /**< Capacity of device memory in bytes */

    /**
     * @brief Reallocates host memory to the specified size.
     *
     * @param bytes Size of host memory to allocate in bytes.
     */
    void ReallocHost(int64_t bytes);

    /**
     * @brief Reallocates device memory to the specified size.
     *
     * @param bytes Size of device memory to allocate in bytes.
     */
    void ReallocDevice(int64_t bytes);

public:
    /**
     * @brief Constructor.
     *
     * @param dtype_bytes Size of the data type in bytes.
     */
    explicit Tensor(size_t dtype_bytes)
        : dtype_bytes_(dtype_bytes) {
    }

    ~Tensor(); /**< Destructor */

    /**
     * @brief Accessor for the pointer to host memory.
     *
     * @return void* Pointer to host memory.
     */
    void* Host() {
        return host_ptr_;
    }

    /**
     * @brief Allocates or resizes host memory to the specified size.
     *
     * @param size Size of host memory to allocate or resize.
     * @return void* Pointer to host memory.
     */
    void* Host(int64_t size);

    /**
     * @brief Accessor for the pointer to device memory.
     *
     * @return void* Pointer to device memory.
     */
    void* Device() {
        return device_ptr_;
    }

    /**
     * @brief Allocates or resizes device memory to the specified size.
     *
     * @param size Size of device memory to allocate or resize.
     * @return void* Pointer to device memory.
     */
    void* Device(int64_t size);

    /**
     * @brief Returns the size of the tensor on the host side.
     *
     * @return int64_t Size of the tensor on the host side.
     */
    [[nodiscard]] int64_t HostSize() const {
        return host_bytes_ / dtype_bytes_;
    }

    /**
     * @brief Returns the size of the tensor on the device side.
     *
     * @return int64_t Size of the tensor on the device side.
     */
    [[nodiscard]] int64_t DeviceSize() const {
        return device_bytes_ / dtype_bytes_;
    }
};

/**
 * @brief Struct representing Tensor information.
 */
struct TensorInfo {
private:
    int32_t index_;            /**< Index of the tensor */

public:
    std::string    name;       /**< Name of the tensor */
    nvinfer1::Dims dims;       /**< Dimensions of the tensor */
    bool           is_dynamic; /**< Whether the tensor is dynamic */
    bool           is_input;   /**< Whether the tensor is an input */
    size_t         type_size;  /**< Type size of the tensor */
    int64_t        vol;        /**< Volume of the tensor */
    Tensor         tensor;     /**< Tensor object */

    /**
     * @brief Constructor.
     *
     * @param index Index of the tensor.
     * @param name Name of the tensor.
     * @param dims Dimensions of the tensor.
     * @param is_dynamic Whether the tensor is dynamic.
     * @param is_input Whether the tensor is an input.
     * @param data_type Data type of the tensor.
     */
    TensorInfo(int32_t index, const char* name, const nvinfer1::Dims& dims,
               bool is_dynamic, bool is_input, nvinfer1::DataType data_type)
        : index_(index),
          name(name),
          dims(dims),
          is_dynamic(is_dynamic),
          is_input(is_input),
          type_size(GetDataTypeSize(data_type)),
          tensor(Tensor(GetDataTypeSize(data_type))),
          vol(CalculateVolume(dims)) {
    }

    /**
     * @brief Accessor for the index of the tensor.
     *
     * @return int32_t Index of the tensor.
     */
    [[nodiscard]] int32_t index() const {
        return index_;
    }

    /**
     * @brief Updates the volume of the tensor.
     */
    void UpdateVolume() {
        vol = CalculateVolume(dims);
    }
};

}  // namespace deploy