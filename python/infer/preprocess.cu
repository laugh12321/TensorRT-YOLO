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

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;

    int y_low = floorf(d2s[3] * dx + d2s[4] * dy + d2s[5] + 0.5f);
    int x_low = floorf(d2s[0] * dx + d2s[1] * dy + d2s[2] + 0.5f);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    int indices[4];
    indices[0] = y_low * src_line_size + x_low * sizeof(uint8_t) * 3;
    indices[1] = y_low * src_line_size + x_high * sizeof(uint8_t) * 3;
    indices[2] = y_high * src_line_size + x_low * sizeof(uint8_t) * 3;
    indices[3] = y_high * src_line_size + x_high * sizeof(uint8_t) * 3;

    uchar3* v1 = reinterpret_cast<uchar3*>(src + indices[0]);
    uchar3* v2 = reinterpret_cast<uchar3*>(src + indices[1]);
    uchar3* v3 = reinterpret_cast<uchar3*>(src + indices[2]);
    uchar3* v4 = reinterpret_cast<uchar3*>(src + indices[3]);

    float ly = d2s[3] * dx + d2s[4] * dy + d2s[5] + 0.5f - y_low;
    float lx = d2s[0] * dx + d2s[1] * dy + d2s[2] + 0.5f - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    // bgr -> rgb
    c0 = w1 * v1->z + w2 * v2->z + w3 * v3->z + w4 * v4->z;
    c1 = w1 * v1->y + w2 * v2->y + w3 * v3->y + w4 * v4->y;
    c2 = w1 * v1->x + w2 * v2->x + w3 * v3->x + w4 * v4->x;

    // 合并归一化操作
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // rgbrgbrgb -> rrrgggbbb
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

    float c0 = fill_value, c1 = fill_value, c2 = fill_value;

    int y_low = floorf(d2s[3] * dx + d2s[4] * dy + d2s[5] + 0.5f);
    int x_low = floorf(d2s[0] * dx + d2s[1] * dy + d2s[2] + 0.5f);
    int y_high = y_low + 1;
    int x_high = x_low + 1;

    int indices[4];
    indices[0] = y_low * src_line_size + x_low * sizeof(uint8_t) * 3;
    indices[1] = y_low * src_line_size + x_high * sizeof(uint8_t) * 3;
    indices[2] = y_high * src_line_size + x_low * sizeof(uint8_t) * 3;
    indices[3] = y_high * src_line_size + x_high * sizeof(uint8_t) * 3;

    uchar3* v1 = reinterpret_cast<uchar3*>(src + indices[0]);
    uchar3* v2 = reinterpret_cast<uchar3*>(src + indices[1]);
    uchar3* v3 = reinterpret_cast<uchar3*>(src + indices[2]);
    uchar3* v4 = reinterpret_cast<uchar3*>(src + indices[3]);

    float ly = d2s[3] * dx + d2s[4] * dy + d2s[5] + 0.5f - y_low;
    float lx = d2s[0] * dx + d2s[1] * dy + d2s[2] + 0.5f - x_low;
    float hy = 1 - ly;
    float hx = 1 - lx;
    float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

    // bgr -> rgb
    c0 = w1 * v1->z + w2 * v2->z + w3 * v3->z + w4 * v4->z;
    c1 = w1 * v1->y + w2 * v2->y + w3 * v3->y + w4 * v4->y;
    c2 = w1 * v1->x + w2 * v2->x + w3 * v3->x + w4 * v4->x;

    // 合并归一化操作
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // rgbrgbrgb -> rrrgggbbb
    int area = dst_width * dst_height;
    float* pdst_c0 = dst + dy * dst_width + dx;
    float* pdst_c1 = pdst_c0 + area;
    float* pdst_c2 = pdst_c1 + area;
    *pdst_c0 = c0;
    *pdst_c1 = c1;
    *pdst_c2 = c2;
}
