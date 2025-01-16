/**
 * @file warpaffine.cu
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 仿射变换矩阵实现以及 CUDA 实现的仿射变换函数
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <cstdint>

#include "deploy/infer/warpaffine.hpp"

namespace deploy {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ float3 operator*(float value0, float3 value1) {
    float3 result;
    result.x = value0 * value1.x;
    result.y = value0 * value1.y;
    result.z = value0 * value1.z;

    return result;
}

__device__ float3 operator+(float3& value0, float3& value1) {
    float3 result;
    result.x = value0.x + value1.x;
    result.y = value0.y + value1.y;
    result.z = value0.z + value1.z;

    return result;
}

__device__ void operator+=(float3& result, float3& value) {
    result.x += value.x;
    result.y += value.y;
    result.z += value.z;
}

__device__ float3 operator+=(float3& value0, const float3& value1) {
    value0.x += value1.x;
    value0.y += value1.y;
    value0.z += value1.z;
    return value0;
}

__device__ float3 uchar3_to_float3(uchar3 v) {
    return make_float3(v.x, v.y, v.z);
}

__device__ bool in_bounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

__device__ void warp_affine_bilinear(const uint8_t* src, const int src_cols, const int src_rows,
                                     float* dst, const int dst_cols, const int dst_rows,
                                     const float3 m0, const float3 m1, const ProcessConfig config, int element_x, int element_y) {
    if (element_x >= dst_cols || element_y >= dst_rows) {
        return;
    }

    float2 src_xy = make_float2(
        m0.x * element_x + m0.y * element_y + m0.z,
        m1.x * element_x + m1.y * element_y + m1.z);

    int src_x0 = __float2int_rd(src_xy.x);
    int src_y0 = __float2int_rd(src_xy.y);
    int src_x1 = src_x0 + 1;
    int src_y1 = src_y0 + 1;

    float wx0 = src_x1 - src_xy.x;
    float wx1 = src_xy.x - src_x0;
    float wy0 = src_y1 - src_xy.y;
    float wy1 = src_xy.y - src_y0;

    float3 src_value0, src_value1, value0, value1;
    bool   flag0 = in_bounds(src_x0, src_y0, src_cols, src_rows);
    bool   flag1 = in_bounds(src_x1, src_y0, src_cols, src_rows);
    bool   flag2 = in_bounds(src_x0, src_y1, src_cols, src_rows);
    bool   flag3 = in_bounds(src_x1, src_y1, src_cols, src_rows);

    float3  border_value = make_float3(config.border_value, config.border_value, config.border_value);
    uchar3* input        = (uchar3*)((uint8_t*)src + src_y0 * src_cols * 3);
    src_value0           = flag0 ? uchar3_to_float3(input[src_x0]) : border_value;
    src_value1           = flag1 ? uchar3_to_float3(input[src_x1]) : border_value;
    value0               = wx0 * wy0 * src_value0;
    value1               = wx1 * wy0 * src_value1;
    float3 sum           = value0 + value1;

    input       = (uchar3*)((uint8_t*)src + src_y1 * src_cols * 3);
    src_value0  = flag2 ? uchar3_to_float3(input[src_x0]) : border_value;
    src_value1  = flag3 ? uchar3_to_float3(input[src_x1]) : border_value;
    value0      = wx0 * wy1 * src_value0;
    value1      = wx1 * wy1 * src_value1;
    sum        += value0 + value1;

    if (config.swap_rb) {
        float temp = sum.x;
        sum.x      = sum.z;
        sum.z      = temp;
    }

    float* output                   = (float*)dst + element_y * dst_cols + element_x;
    output[0]                       = sum.x * config.alpha.x + config.beta.x;
    output[dst_cols * dst_rows]     = sum.y * config.alpha.y + config.beta.y;
    output[2 * dst_cols * dst_rows] = sum.z * config.alpha.z + config.beta.z;
}

__global__ void gpuBilinearWarpAffine(const void* src, const int src_cols, const int src_rows,
                                      void* dst, const int dst_cols, const int dst_rows,
                                      const float3 m0, const float3 m1, const ProcessConfig config) {
    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    warp_affine_bilinear(static_cast<const uint8_t*>(src), src_cols, src_rows,
                         static_cast<float*>(dst), dst_cols, dst_rows,
                         m0, m1, config,
                         element_x, element_y);
}

__global__ void gpuMutliBilinearWarpAffine(const void* src, const int src_cols, const int src_rows,
                                           void* dst, const int dst_cols, const int dst_rows,
                                           const float3 m0, const float3 m1, const ProcessConfig config,
                                           int num_images) {
    int image_idx = blockIdx.z;
    if (image_idx >= num_images) {
        return;
    }

    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    warp_affine_bilinear(static_cast<const uint8_t*>(src) + image_idx * src_rows * src_cols * 3,
                         src_cols, src_rows,
                         static_cast<float*>(dst) + image_idx * dst_rows * dst_cols * 3,
                         dst_cols, dst_rows,
                         m0, m1, config,
                         element_x, element_y);
}

void AffineTransform::updateMatrix(int src_width, int src_height, int dst_width, int dst_height) {
    if (src_width == last_src_width_ && src_height == last_src_height_) return;
    last_src_width_  = src_width;
    last_src_height_ = src_height;

    double scale  = std::min(static_cast<double>(dst_width) / src_width, static_cast<double>(dst_height) / src_height);
    double offset = 0.5 * scale - 0.5;

    double scale_from_width  = -0.5 * scale * src_width;
    double scale_from_height = -0.5 * scale * src_height;
    double half_dst_width    = 0.5 * dst_width;
    double half_dst_height   = 0.5 * dst_height;

    double inv_d = (scale != 0.0) ? 1.0 / (scale * scale) : 0.0;
    double a     = scale * inv_d;

    matrix[0] = make_float3(a, 0.0, -a * (scale_from_width + half_dst_width + offset));
    matrix[1] = make_float3(0.0, a, -a * (scale_from_height + half_dst_height + offset));

    dst_offset_x = int(dst_width * 0.5 + scale_from_width);
    dst_offset_y = int(dst_height * 0.5 + scale_from_height);
}

void AffineTransform::applyTransform(float x, float y, float* transformed_x, float* transformed_y) const {
    *transformed_x = matrix[0].x * x + matrix[0].y * y + matrix[0].z;
    *transformed_y = matrix[1].x * x + matrix[1].y * y + matrix[1].z;
}

void cudaWarpAffine(const void* src, const int src_cols, const int src_rows,
                    void* dst, const int dst_cols, const int dst_rows,
                    const float3 matrix[2], const ProcessConfig config, cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(dst_cols, blockDim.x), iDivUp(dst_rows, blockDim.y));
    gpuBilinearWarpAffine<<<gridDim, blockDim, 0, stream>>>(src, src_cols, src_rows, dst, dst_cols, dst_rows, matrix[0], matrix[1], config);
}

void cudaMutliWarpAffine(const void* src, const int src_cols, const int src_rows,
                         void* dst, const int dst_cols, const int dst_rows,
                         const float3 matrix[2], const ProcessConfig config, int num_images, cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(dst_cols, blockDim.x), iDivUp(dst_rows, blockDim.y), num_images);
    gpuMutliBilinearWarpAffine<<<gridDim, blockDim, 0, stream>>>(src, src_cols, src_rows, dst, dst_cols, dst_rows, matrix[0], matrix[1], config, num_images);
}

}  // namespace deploy
