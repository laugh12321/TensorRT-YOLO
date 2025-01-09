/**
 * @file warpaffine.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义仿射变换矩阵以及仿射变换函数
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cuda_runtime_api.h>

#include "deploy/core/option.hpp"

namespace deploy {

/**
 * @brief 用于仿射变换的 2x3 变换矩阵的结构体
 *
 */
struct AffineTransform {
    float3 matrix[2];         // < 用于仿射变换的 2x3 变换矩阵
    int    dst_offset_x;      // < 变换后目标图像的 X 轴偏移量。
    int    dst_offset_y;      // < 变换后目标图像的 Y 轴偏移量。
    int    last_src_width_;   // < 上一次处理的源图像的宽度。
    int    last_src_height_;  // < 上一次处理的源图像的高度。

    /**
     * @brief 根据源图像和目标图像尺寸的变化更新仿射变换矩阵
     *
     * @param src_width 源图像的宽度
     * @param src_height 源图像的高度
     * @param dst_width 目标图像的宽度
     * @param dst_height 目标图像的高度
     */
    void updateMatrix(int src_width, int src_height, int dst_width, int dst_height);

    /**
     * @brief 使用仿射变换矩阵变换一个点
     *
     * @param x 点的 X 坐标
     * @param y 点的 Y 坐标
     * @param[out] transformed_x 变换后的 X 坐标
     * @param[out] transformed_y 变换后的 Y 坐标
     */
    void applyTransform(float x, float y, float* transformed_x, float* transformed_y) const;
};

/**
 * @brief 使用 CUDA 应用仿射变换。
 *
 * @param src 输入图像数据的指针。
 * @param src_cols 输入图像的宽度。
 * @param src_rows 输入图像的高度。
 * @param dst 输出图像数据的指针。
 * @param dst_cols 输出图像的宽度。
 * @param dst_rows 输出图像的高度。
 * @param matrix 仿射变换矩阵。
 * @param config 处理配置参数。
 * @param stream 用于异步执行的 CUDA 流。
 */
void cudaWarpAffine(const void* src, const int src_cols, const int src_rows,
                    void* dst, const int dst_cols, const int dst_rows,
                    const float3 matrix[2], const ProcessConfig config, cudaStream_t stream);

}  // namespace deploy