/**
 * @file letterbox.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义等比例缩放并填充图像，保持原始比例的函数
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cuda_runtime_api.h>

#include "utils/common.hpp"

namespace trtyolo {

/**
 * @brief 用于仿射变换的 2x3 变换矩阵的结构体
 *
 */
struct Transform {
    int4  meta;              // < 变换元数据，包含有效宽度、有效高度、偏移量 X、偏移量 Y
    float scale;             // < 缩放比例
    int   last_src_width_;   // < 上一次处理的源图像的宽度。
    int   last_src_height_;  // < 上一次处理的源图像的高度。

    /**
     * @brief 根据源图像和目标图像尺寸的变化更新仿射变换矩阵
     *
     * @param src_width 源图像的宽度
     * @param src_height 源图像的高度
     * @param dst_width 目标图像的宽度
     * @param dst_height 目标图像的高度
     */
    void update(int src_width, int src_height, int dst_width, int dst_height);

    /**
     * @brief 使用仿射变换矩阵变换一个点
     *
     * @param x 点的 X 坐标
     * @param y 点的 Y 坐标
     * @param[out] transformed_x 变换后的 X 坐标
     * @param[out] transformed_y 变换后的 Y 坐标
     */
    void apply(float x, float y, float* transformed_x, float* transformed_y) const;
};

/**
 * @brief 使用 CUDA 对图像应用 Letterbox 缩放和归一化。
 *
 * @param src 输入图像数据的指针
 * @param src_cols 输入图像的宽度
 * @param src_rows 输入图像的高度
 * @param src_pitch 输入图像的行步长（每行字节数），用于处理不同宽度的图像
 * @param dst 输出图像数据的指针
 * @param dst_cols 输出图像的宽度
 * @param dst_rows 输出图像的高度
 * @param meta 变换元数据，包含有效宽度、有效高度、偏移量 X、偏移量 Y
 * @param config 处理配置参数
 * @param stream 用于异步执行的 CUDA 流
 */
void cudaLetterbox(const void* src, const int src_cols, const int src_rows, const size_t src_pitch,
                   void* dst, const int dst_cols, const int dst_rows, const int4 meta,
                   const ProcessConfig config, cudaStream_t stream);

/**
 * @brief 使用 CUDA 对多张图像应用 Letterbox 缩放和归一化。
 *
 * 此函数对多张输入图像同时应用 Letterbox 变换，将变换后的图像输出到指定的内存位置
 * 与 `cudaLetterbox` 类似，但支持批量处理多张图像，提高处理效率
 *
 * @param src 输入图像数据的指针数组，每个元素指向一张输入图像的首地址
 * @param src_cols 输入图像的宽度
 * @param src_rows 输入图像的高度
 * @param src_pitch 输入图像的行步长（每行字节数），用于处理不同宽度的图像
 * @param dst 输出图像数据的指针数组，每个元素指向一张输出图像的首地址
 * @param dst_cols 输出图像的宽度
 * @param dst_rows 输出图像的高度
 * @param meta 变换元数据，包含有效宽度、有效高度、偏移量 X、偏移量 Y
 * @param config 处理配置参数，如插值方法、边界处理等
 * @param num_images 图像数量，即 `src` 和 `dst` 数组的长度
 * @param stream 用于异步执行的 CUDA 流
 */
void cudaMultiLetterbox(const void* src, const int src_cols, const int src_rows, const size_t src_pitch,
                        void* dst, const int dst_cols, const int dst_rows, const int4 meta,
                        const ProcessConfig config, int num_images, cudaStream_t stream);

}  // namespace trtyolo