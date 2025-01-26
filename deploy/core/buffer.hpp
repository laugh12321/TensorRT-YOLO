/**
 * @file buffer.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义了用于管理内存操作的抽象基类 Buffer 和几种具体的 Buffer 类型
 * @date 2025-01-08
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <numeric>
#include <string>

namespace deploy {

/**
 * @brief 抽象基类 Buffer，用于管理内存操作
 *
 */
class BaseBuffer {
public:
    virtual ~BaseBuffer() = default;

    /**
     * @brief 分配内存
     *
     * @param size 要分配的内存大小
     */
    virtual void allocate(size_t size) = 0;

    /**
     * @brief 释放内存
     *
     */
    virtual void free() = 0;

    /**
     * @brief 获取设备内存指针
     *
     * @return void* 设备内存指针
     */
    virtual void* device() = 0;

    /**
     * @brief 获取主机内存指针
     *
     * @return void* 主机内存指针
     */
    virtual void* host() = 0;

    /**
     * @brief 获取内存大小
     *
     * @return size_t 内存大小
     */
    virtual size_t size() const = 0;

    /**
     * @brief 从主机到设备拷贝数据
     *
     * @param stream CUDA流
     */
    virtual void hostToDevice(cudaStream_t stream = nullptr) = 0;

    /**
     * @brief 从设备到主机拷贝数据
     *
     * @param stream CUDA流
     */
    virtual void deviceToHost(cudaStream_t stream = nullptr) = 0;
};

/**
 * @brief DeviceBuffer 类，表示设备内存
 *
 */
class DeviceBuffer : public BaseBuffer {
public:
    DeviceBuffer() : size_(0), device_(nullptr) {}
    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
    ~DeviceBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  device_;  // < 设备内存指针
    size_t size_;    // < 内存大小
};

/**
 * @brief DiscreteBuffer 类，表示具有主机和设备内存的分离内存
 *
 */
class DiscreteBuffer : public BaseBuffer {
public:
    DiscreteBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    DiscreteBuffer(const DiscreteBuffer&)            = delete;
    DiscreteBuffer& operator=(const DiscreteBuffer&) = delete;
    DiscreteBuffer(DiscreteBuffer&& other) noexcept;
    DiscreteBuffer& operator=(DiscreteBuffer&& other) noexcept;
    ~DiscreteBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;    // < 主机内存指针
    void*  device_;  // < 设备内存指针
    size_t size_;    // < 内存大小
};

/**
 * @brief UnifiedBuffer 类，表示统一内存（主机和设备共享内存）
 *
 */
class UnifiedBuffer : public BaseBuffer {
public:
    UnifiedBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    UnifiedBuffer(const UnifiedBuffer&)            = delete;
    UnifiedBuffer& operator=(const UnifiedBuffer&) = delete;
    UnifiedBuffer(UnifiedBuffer&& other) noexcept;
    UnifiedBuffer& operator=(UnifiedBuffer&& other) noexcept;
    ~UnifiedBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;    // < 主机内存指针
    void*  device_;  // < 设备内存指针
    size_t size_;    // < 内存大小
};

/**
 * @brief MappedBuffer 类，表示映射内存（主机和设备共享映射内存）
 *
 */
class MappedBuffer : public BaseBuffer {
public:
    MappedBuffer() : size_(0), host_(nullptr), device_(nullptr) {}
    MappedBuffer(const MappedBuffer&)            = delete;
    MappedBuffer& operator=(const MappedBuffer&) = delete;
    MappedBuffer(MappedBuffer&& other) noexcept;
    MappedBuffer& operator=(MappedBuffer&& other) noexcept;
    ~MappedBuffer() { free(); }

    void   allocate(size_t size) override;
    void   free() override;
    void*  device() override;
    void*  host() override;
    size_t size() const override;
    void   hostToDevice(cudaStream_t stream = nullptr) override;
    void   deviceToHost(cudaStream_t stream = nullptr) override;

private:
    void*  host_;    // < 主机内存指针
    void*  device_;  // < 设备内存指针
    size_t size_;    // < 内存大小
};

/**
 * @brief Buffer 类型枚举，用于选择不同类型的 Buffer
 *
 */
enum class BufferType {
    Device,    // < 设备内存
    Discrete,  // < 分离内存（主机和设备都有内存）
    Unified,   // < 统一内存（设备和主机共享内存）
    Mapped     // < 映射内存（用于NVIDIA集成设备）
};

/**
 * @brief Buffer 工厂类，根据类型创建不同的 Buffer
 *
 */
class BufferFactory {
public:
    /**
     * @brief 创建指定类型的 Buffer
     *
     * @param type 要创建的 Buffer 类型
     * @return 指定类型的 Buffer 智能指针
     */
    static std::unique_ptr<BaseBuffer> createBuffer(BufferType type);
};

/**
 * @brief TensorInfo 结构体，表示张量的信息
 *
 */
struct TensorInfo {
private:
    nvinfer1::DataType dtype_;           // < 张量数据类型
    size_t             bytes_;           // < 张量大小（字节数）

public:
    std::string                 name;    // < 张量名称
    nvinfer1::Dims              shape;   // < 张量形状
    bool                        input;   // < 是否为输入张量
    std::unique_ptr<BaseBuffer> buffer;  // < 张量对应的内存

    /**
     * @brief 构造函数，初始化张量信息
     *
     * @param name 张量名称
     * @param shape 张量形状
     * @param dtype 张量数据类型
     * @param input 是否为输入张量
     */
    TensorInfo(const std::string name, const nvinfer1::Dims& shape, const nvinfer1::DataType dtype, const bool input, BufferType buffer_type)
        : name(name), shape(shape), dtype_(dtype), input(input) {
        buffer = BufferFactory::createBuffer(buffer_type);
        update();
    }

    /**
     * @brief 更新张量的大小并分配内存
     *
     */
    void update() {
        bytes_ = std::accumulate(shape.d, shape.d + shape.nbDims, 1, std::multiplies<int>()) * dtype_to_bytes(dtype_);
        buffer->allocate(bytes_);
    }

private:
    /**
     * @brief 将数据类型转换为字节数
     *
     * @param dtype 数据类型
     * @return size_t 数据类型对应的字节数
     */
    size_t dtype_to_bytes(nvinfer1::DataType dtype) {
        switch (dtype) {
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
            default:
                break;
        }
        return 0;
    }
};

}  // namespace deploy
