#include "deploy/vision/cudaWarp.hpp"

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

__device__ bool inBounds(int x, int y, int cols, int rows) {
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

__global__ void gpuBilinearWarpAffine(const uint8_t* src, const int src_cols, const int src_rows,
                                      float* dst, const int dst_cols, const int dst_rows,
                                      const float3 m0, const float3 m1) {
    int element_x = blockDim.x * blockIdx.x + threadIdx.x;
    int element_y = blockDim.y * blockIdx.y + threadIdx.y;
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
    bool   flag0 = inBounds(src_x0, src_y0, src_cols, src_rows);
    bool   flag1 = inBounds(src_x1, src_y0, src_cols, src_rows);
    bool   flag2 = inBounds(src_x0, src_y1, src_cols, src_rows);
    bool   flag3 = inBounds(src_x1, src_y1, src_cols, src_rows);

    float3  border_value = make_float3(114.0f, 114.0f, 114.0f);
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

    float* output                   = (float*)dst + element_y * dst_cols + element_x;
    output[0]                       = sum.x * 0.00392156862f;
    output[dst_cols * dst_rows]     = sum.y * 0.00392156862f;
    output[2 * dst_cols * dst_rows] = sum.z * 0.00392156862f;
}

void TransformMatrix::update(int fromWidth, int fromHeight, int toWidth, int toHeight) {
    if (fromWidth == lastWidth && fromHeight == lastHeight) return;
    lastWidth  = fromWidth;
    lastHeight = fromHeight;

    double scale  = std::min(static_cast<double>(toWidth) / fromWidth, static_cast<double>(toHeight) / fromHeight);
    double offset = 0.5 * scale - 0.5;

    double scaleFromWidth  = -0.5 * scale * fromWidth;
    double scaleFromHeight = -0.5 * scale * fromHeight;
    double halfToWidth     = 0.5 * toWidth;
    double halfToHeight    = 0.5 * toHeight;

    double invD = (scale != 0.0) ? 1.0 / (scale * scale) : 0.0;
    double A    = scale * invD;

    matrix[0] = make_float3(A, 0.0, -A * (scaleFromWidth + halfToWidth + offset));
    matrix[1] = make_float3(0.0, A, -A * (scaleFromHeight + halfToHeight + offset));

    dw = int(toWidth * 0.5 + scaleFromWidth);
    dh = int(toHeight * 0.5 + scaleFromHeight);
}

void TransformMatrix::transform(float x, float y, float* ox, float* oy) const {
    *ox = matrix[0].x * x + matrix[0].y * y + matrix[0].z;
    *oy = matrix[1].x * x + matrix[1].y * y + matrix[1].z;
}

void cudaWarpAffine(const uint8_t* src, const int src_cols, const int src_rows,
                    float* dst, const int dst_cols, const int dst_rows,
                    const float3 matrix[2], cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(16, 16);
    const dim3 gridDim(iDivUp(dst_cols, blockDim.x), iDivUp(dst_rows, blockDim.y));
    gpuBilinearWarpAffine<<<gridDim, blockDim, 0, stream>>>(src, src_cols, src_rows, dst, dst_cols, dst_rows, matrix[0], matrix[1]);
}

}  // namespace deploy
