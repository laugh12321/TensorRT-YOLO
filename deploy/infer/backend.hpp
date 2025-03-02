/**
 * @file backend.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief TensorRT 推理后端定义
 * @date 2025-01-15
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <NvInferRuntime.h>

#include <memory>
#include <string>
#include <vector>

#include "deploy/core/buffer.hpp"
#include "deploy/core/core.hpp"
#include "deploy/infer/warpaffine.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"

namespace deploy {

/**
 * @brief TensorRT 后端类，用于执行推理操作。
 */
class DEPLOYAPI TrtBackend {
public:
    /**
     * @brief 构造函数，用于初始化 TrtBackend 对象。
     *
     * @param trt_engine_file TensorRT 引擎文件路径。
     * @param infer_option 推理选项指针。
     */
    TrtBackend(const std::string& trt_engine_file, const InferOption& infer_option);

    /**
     * @brief 默认构造函数。
     */
    TrtBackend() = default;

    /**
     * @brief 析构函数。
     */
    ~TrtBackend();

    /**
     * @brief 克隆 TrtBackend 对象。
     *
     * @return 克隆后的 TrtBackend 对象的智能指针。
     */
    std::unique_ptr<TrtBackend> clone();

    /**
     * @brief 执行推理操作。
     *
     * @param inputs 输入图像向量。
     */
    void infer(const std::vector<Image>& inputs);

    cudaStream_t                 stream;             // < CUDA 流
    InferOption                  option;             // < 推理选项
    std::vector<TensorInfo>      tensor_infos;       // < 张量信息向量
    std::vector<AffineTransform> affine_transforms;  // < 仿射变换向量
    int4                         min_shape;          // < 最小形状
    int4                         max_shape;          // < 最大形状
    bool                         dynamic;            // < 是否为动态形状

private:
    void getTensorInfo();
    void initialize();
    void captureCudaGraph();
    void dynamicInfer(const std::vector<Image>& inputs);
    void staticInfer(const std::vector<Image>& inputs);

    std::unique_ptr<TRTManager> manager_;        // < TensorRT 管理器对象的智能指针
    CudaGraph                   cuda_graph_;     // < CUDA 图
    std::unique_ptr<BaseBuffer> inputs_buffer_;  // < 输入缓冲区智能指针
    BufferType                  buffer_type_;    // < 缓冲区类型

    bool zero_copy_;                             // < 是否为零拷贝

    int input_size_;                             // < 输入大小
    int infer_size_;                             // < 推理大小
};

}  // namespace deploy