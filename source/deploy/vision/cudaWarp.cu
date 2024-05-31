#include <algorithm>

#include "deploy/vision/cudaWarp.hpp"

namespace deploy {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void gpuPerspectiveWarp(uint8_t* input, int inputWidth, int inputHeight,
                                   float* output, int outputWidth, int outputHeight,
                                   float3 m0, float3 m1, float3 m2) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= outputWidth || y >= outputHeight)
        return;

    const float3 vec = make_float3(x, y, 1.0f);

    const float3 vecOut = make_float3(m0.x * vec.x + m0.y * vec.y + m0.z * vec.z,
                                      m1.x * vec.x + m1.y * vec.y + m1.z * vec.z,
                                      m2.x * vec.x + m2.y * vec.y + m2.z * vec.z);

    const int u = __float2int_rd(vecOut.x / vecOut.z);
    const int v = __float2int_rd(vecOut.y / vecOut.z);

    const int index = y * outputWidth + x;
    const int area = outputWidth * outputHeight;

    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

    // Use texture memory for better caching if the input is read-only
    if (u >= 0 && u < inputWidth && v >= 0 && v < inputHeight) {
        const int idx = (v * inputWidth + u) * 3;

        c0 = input[idx] / 255.0f;
        c1 = input[idx + 1] / 255.0f;
        c2 = input[idx + 2] / 255.0f;
    }

    output[index] = c0;
    output[index + area] = c1;
    output[index + 2 * area] = c2;

    // without channel swap
    // output[index * 3] = c0;
    // output[index * 3 + 1] = c1;
    // output[index * 3 + 2] = c2;
}

void TransformMatrix::update(int fromWidth, int fromHeight, int toWidth, int toHeight) {
    if (fromWidth == lastWidth && fromHeight == lastHeight) return;
    lastWidth = fromWidth;
    lastHeight = fromHeight;

    double scale = std::min(static_cast<double>(toWidth) / fromWidth, static_cast<double>(toHeight) / fromHeight);
    double offset = 0.5 * scale - 0.5;

    double scaleFromWidth = -0.5 * scale * fromWidth;
    double scaleFromHeight = -0.5 * scale * fromHeight;
    double halfToWidth = 0.5 * toWidth;
    double halfToHeight = 0.5 * toHeight;

    double invD = (scale != 0.0) ? 1.0 / (scale * scale) : 0.0;
    double A = scale * invD;

    matrix[0] = make_float3(A, 0.0, -A * (scaleFromWidth + halfToWidth + offset));
    matrix[1] = make_float3(0.0, A, -A * (scaleFromHeight + halfToHeight + offset));
    matrix[2] = make_float3(0.0, 0.0, 1.0);
}

void TransformMatrix::transform(float x, float y, float* ox, float* oy) const {
    *ox = matrix[0].x * x + matrix[0].y * y + matrix[0].z;
    *oy = matrix[1].x * x + matrix[1].y * y + matrix[1].z;
}

void cudaWarpAffine(uint8_t* input, uint32_t inputWidth, uint32_t inputHeight,
                    float* output, uint32_t outputWidth, uint32_t outputHeight,
                    float3 matrix[3], cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y));
    gpuPerspectiveWarp<<<gridDim, blockDim, 0, stream>>>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, matrix[0], matrix[1], matrix[2]);
}


}  // namespace deploy
