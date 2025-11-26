/**
 * @file letterbox.cu
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 等比例缩放并填充图像，保持原始比例
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <cstdint>

#include "letterbox.hpp"

namespace trtyolo {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__device__ void letterbox(
    const uint8_t* __restrict__ src, const int src_cols, const int src_rows, const size_t src_pitch,
    float* __restrict__ dst, const int dst_cols, const int dst_rows, const int4 meta,
    const ProcessConfig config, int element_x, int element_y) {
    const int base_x = element_x << 1;  // *2
    const int base_y = element_y << 1;  // *2
    if (base_y >= dst_rows) return;

    const int valid_w  = meta.x;
    const int valid_h  = meta.y;
    const int offset_x = meta.z;
    const int offset_y = meta.w;

    const float scale_x = src_cols * __frcp_rn(valid_w);
    const float scale_y = src_rows * __frcp_rn(valid_h);
    const int   vx_end  = offset_x + valid_w;
    const int   vy_end  = offset_y + valid_h;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
        int py = base_y + dy;
        if (py >= dst_rows) continue;

#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            int px = base_x + dx;
            if (px >= dst_cols) continue;

            const bool in_roi = !((px < offset_x) | (px >= vx_end) | (py < offset_y) | (py >= vy_end));
            float3     pixel  = make_float3(config.border_value, config.border_value, config.border_value);

            if (in_roi) {
                const float sx = __fmaf_rn(px - offset_x + 0.5f, scale_x, -0.5f);
                const float sy = __fmaf_rn(py - offset_y + 0.5f, scale_y, -0.5f);

                const int x0 = max(0, min(__float2int_rd(sx), src_cols - 2));
                const int y0 = max(0, min(__float2int_rd(sy), src_rows - 2));
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;

                const float a   = sx - x0;
                const float b   = sy - y0;
                const float w00 = (1.0f - a) * (1.0f - b);
                const float w10 = a * (1.0f - b);
                const float w01 = (1.0f - a) * b;
                const float w11 = a * b;

                const uint8_t* __restrict__ p00 = src + y0 * src_pitch + x0 * 3;
                const uint8_t* __restrict__ p10 = src + y0 * src_pitch + x1 * 3;
                const uint8_t* __restrict__ p01 = src + y1 * src_pitch + x0 * 3;
                const uint8_t* __restrict__ p11 = src + y1 * src_pitch + x1 * 3;

                pixel.x = fmaf(w00, __ldg(p00), fmaf(w10, __ldg(p10), fmaf(w01, __ldg(p01), w11 * __ldg(p11))));
                pixel.y = fmaf(w00, __ldg(p00 + 1), fmaf(w10, __ldg(p10 + 1), fmaf(w01, __ldg(p01 + 1), w11 * __ldg(p11 + 1))));
                pixel.z = fmaf(w00, __ldg(p00 + 2), fmaf(w10, __ldg(p10 + 2), fmaf(w01, __ldg(p01 + 2), w11 * __ldg(p11 + 2))));
            }

            if (config.swap_rb) {
                float temp = pixel.x;
                pixel.x    = pixel.z;
                pixel.z    = temp;
            }

            float* out                   = (float*)dst + py * dst_cols + px;
            out[0]                       = pixel.x * config.alpha.x + config.beta.x;
            out[dst_cols * dst_rows]     = pixel.y * config.alpha.y + config.beta.y;
            out[2 * dst_cols * dst_rows] = pixel.z * config.alpha.z + config.beta.z;
        }
    }
}

__global__ void gpuLetterbox(const void* __restrict__ src, const int src_cols, const int src_rows, const size_t src_pitch,
                             void* __restrict__ dst, const int dst_cols, const int dst_rows, const int4 meta, const ProcessConfig config) {
    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    letterbox(static_cast<const uint8_t*>(src), src_cols, src_rows, src_pitch,
              static_cast<float*>(dst), dst_cols, dst_rows, meta, config,
              element_x, element_y);
}

__global__ void gpuMutliLetterbox(const void* __restrict__ src, const int src_cols, const int src_rows, const size_t src_pitch,
                                  void* __restrict__ dst, const int dst_cols, const int dst_rows, const int4 meta, const ProcessConfig config,
                                  int num_images) {
    int image_idx = blockIdx.z;
    if (image_idx >= num_images) return;

    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;

    letterbox(static_cast<const uint8_t*>(src) + image_idx * src_rows * src_pitch,
              src_cols, src_rows, src_pitch,
              static_cast<float*>(dst) + image_idx * dst_rows * dst_cols * 3,
              dst_cols, dst_rows,
              meta, config,
              element_x, element_y);
}

void Transform::update(int src_width, int src_height, int dst_width, int dst_height) {
    if (src_width == last_src_width_ && src_height == last_src_height_) return;
    last_src_width_  = src_width;
    last_src_height_ = src_height;

    scale        = std::min(static_cast<float>(dst_width) / src_width, static_cast<float>(dst_height) / src_height);
    int valid_w  = static_cast<int>(roundf(scale * src_width));
    int valid_h  = static_cast<int>(roundf(scale * src_height));
    int offset_x = (dst_width - valid_w) / 2;
    int offset_y = (dst_height - valid_h) / 2;
    meta         = make_int4(valid_w, valid_h, offset_x, offset_y);
}

void Transform::apply(float x, float y, float* transformed_x, float* transformed_y) const {
    *transformed_x = (x - meta.z) / scale;
    *transformed_y = (y - meta.w) / scale;
}

void cudaLetterbox(const void* src, const int src_cols, const int src_rows, const size_t src_pitch,
                   void* dst, const int dst_cols, const int dst_rows,
                   const int4 meta, const ProcessConfig config, cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(dst_cols, blockDim.x * 2), iDivUp(dst_rows, blockDim.y * 2));
    gpuLetterbox<<<gridDim, blockDim, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, meta, config);
}

void cudaMultiLetterbox(const void* src, const int src_cols, const int src_rows, const size_t src_pitch,
                        void* dst, const int dst_cols, const int dst_rows,
                        const int4 meta, const ProcessConfig config, int num_images, cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(dst_cols, blockDim.x * 2), iDivUp(dst_rows, blockDim.y * 2), num_images);
    gpuMutliLetterbox<<<gridDim, blockDim, 0, stream>>>(src, src_cols, src_rows, src_pitch, dst, dst_cols, dst_rows, meta, config, num_images);
}

}  // namespace trtyolo
