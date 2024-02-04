#include <cstdint>
#include <cuda_fp16.h>

#define _X  ( threadIdx.x + blockIdx.x * blockDim.x )
#define _Y  ( threadIdx.y + blockIdx.y * blockDim.y )


extern "C" __global__ void preprocess_kernel_fp16(uint8_t* src, int src_line_size,
                                  int src_width, int src_height, half* dst,
                                  int dst_width, int dst_height,
                                  uint8_t fill_value, const float* d2s) {
    int dx = _X, dy = _Y;
    if (dx >= dst_width || dy >= dst_height) return;

    // 使用寄存器存储临时变量
    float src_x = d2s[0] * dx + d2s[1] * dy + d2s[2] + 0.5f;
    float src_y = d2s[3] * dx + d2s[4] * dy + d2s[5] + 0.5f;
    float c0 = fill_value, c1 = fill_value, c2 = fill_value;

    if (dx >= 0 && dy >= 0 && dx < dst_width && dy < dst_height) {


        // 计算权重
        int x_low = max(0, min(static_cast<int>(floorf(src_x)), src_width - 1));
        int y_low = max(0, min(static_cast<int>(floorf(src_y)), src_height - 1));
        int x_high = min(src_width - 1, x_low + 1);
        int y_high = min(src_height - 1, y_low + 1);

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        // 直接访问像素值，不使用指针
        uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
        uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
        uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
        uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

        // 权重和计算，不使用分支
        c0 = fmaf(w1, v1[0], fmaf(w2, v2[0], fmaf(w3, v3[0], w4 * v4[0])));
        c1 = fmaf(w1, v1[1], fmaf(w2, v2[1], fmaf(w3, v3[1], w4 * v4[1])));
        c2 = fmaf(w1, v1[2], fmaf(w2, v2[2], fmaf(w3, v3[2], w4 * v4[2])));
    }
    // gbr -> rgb
    float temp = c2;
    c2 = c0;
    c0 = temp;

    // 归一化
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // rgbrgbrgb 到 rrrgggbbb
    int area = dst_width * dst_height;
    half* pdst_c0 = dst + dy * dst_width + dx;
    half* pdst_c1 = pdst_c0 + area;
    half* pdst_c2 = pdst_c1 + area;

    *pdst_c0 = __float2half(c0);
    *pdst_c1 = __float2half(c1);
    *pdst_c2 = __float2half(c2);
}


extern "C" __global__ void preprocess_kernel_fp32(uint8_t* src, int src_line_size,
                                  int src_width, int src_height, float* dst,
                                  int dst_width, int dst_height,
                                  uint8_t fill_value, const float* d2s) {
    int dx = _X, dy = _Y;
    if (dx >= dst_width || dy >= dst_height) return;

    // 使用寄存器存储临时变量
    float src_x = d2s[0] * dx + d2s[1] * dy + d2s[2] + 0.5f;
    float src_y = d2s[3] * dx + d2s[4] * dy + d2s[5] + 0.5f;
    float c0 = fill_value, c1 = fill_value, c2 = fill_value;

    if (dx >= 0 && dy >= 0 && dx < dst_width && dy < dst_height) {


        // 计算权重
        int x_low = max(0, min(static_cast<int>(floorf(src_x)), src_width - 1));
        int y_low = max(0, min(static_cast<int>(floorf(src_y)), src_height - 1));
        int x_high = min(src_width - 1, x_low + 1);
        int y_high = min(src_height - 1, y_low + 1);

        float ly = src_y - y_low;
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;

        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

        // 直接访问像素值，不使用指针
        uint8_t* v1 = src + y_low * src_line_size + x_low * 3;
        uint8_t* v2 = src + y_low * src_line_size + x_high * 3;
        uint8_t* v3 = src + y_high * src_line_size + x_low * 3;
        uint8_t* v4 = src + y_high * src_line_size + x_high * 3;

        // 权重和计算，不使用分支
        c0 = fmaf(w1, v1[0], fmaf(w2, v2[0], fmaf(w3, v3[0], w4 * v4[0])));
        c1 = fmaf(w1, v1[1], fmaf(w2, v2[1], fmaf(w3, v3[1], w4 * v4[1])));
        c2 = fmaf(w1, v1[2], fmaf(w2, v2[2], fmaf(w3, v3[2], w4 * v4[2])));
    }
    // gbr -> rgb
    float temp = c2;
    c2 = c0;
    c0 = temp;

    // 归一化
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // rgbrgbrgb 到 rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;

    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}
