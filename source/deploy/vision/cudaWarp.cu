#include <algorithm>

#include "deploy/vision/cudaWarp.hpp"

namespace deploy {

inline __device__ __host__ int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

__global__ void gpuBilinearWarpAffine(uint8_t* input, int inputWidth, int inputHeight,
                                      float* output, int outputWidth, int outputHeight,
                                      float3 m0, float3 m1) {
    const int x             = blockDim.x * blockIdx.x + threadIdx.x;
    const int y             = blockDim.y * blockIdx.y + threadIdx.y;
    const int inputLineSize = inputWidth * 3;
    const int outputArea    = outputWidth * outputHeight;

    if (x >= outputWidth || y >= outputHeight)
        return;

    float inputX = m0.x * x + m0.y * y + m0.z;
    float inputY = m1.x * x + m1.y * y + m1.z;

    // Initialize to constant value for out of range
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

    // Precompute interpolation coefficients and boundary checks
    if (inputX > -1 && inputX < inputWidth && inputY > -1 && inputY < inputHeight) {
        int lowX  = __float2int_rd(inputX);
        int lowY  = __float2int_rd(inputY);
        int highX = lowX + 1;
        int highY = lowY + 1;

        // Clamp coordinates within image boundaries
        lowX  = max(0, min(lowX, inputWidth - 1));
        highX = max(0, min(highX, inputWidth - 1));
        lowY  = max(0, min(lowY, inputHeight - 1));
        highY = max(0, min(highY, inputHeight - 1));

        // Calculate interpolation weights
        float lx = inputX - lowX;
        float ly = inputY - lowY;
        float hx = 1.0f - lx;
        float hy = 1.0f - ly;

        // Calculate pixel pointers
        uint8_t* v1 = input + lowY * inputLineSize + lowX * 3;
        uint8_t* v2 = input + lowY * inputLineSize + highX * 3;
        uint8_t* v3 = input + highY * inputLineSize + lowX * 3;
        uint8_t* v4 = input + highY * inputLineSize + highX * 3;

        // Perform bilinear interpolation for each channel
        c0 = hy * (hx * v1[0] + lx * v2[0]) + ly * (hx * v3[0] + lx * v4[0]);
        c1 = hy * (hx * v1[1] + lx * v2[1]) + ly * (hx * v3[1] + lx * v4[1]);
        c2 = hy * (hx * v1[2] + lx * v2[2]) + ly * (hx * v3[2] + lx * v4[2]);
    }

    // Normalize values to range [0, 1]
    c0 *= 0.00392156862f;  // Equivalent to c0 /= 255.0f;
    c1 *= 0.00392156862f;
    c2 *= 0.00392156862f;

    // Reorder RGB to RRRGGGBBB
    int index                      = y * outputWidth + x;
    output[index]                  = c0;
    output[index + outputArea]     = c1;
    output[index + 2 * outputArea] = c2;
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
}

void TransformMatrix::transform(float x, float y, float* ox, float* oy) const {
    *ox = matrix[0].x * x + matrix[0].y * y + matrix[0].z;
    *oy = matrix[1].x * x + matrix[1].y * y + matrix[1].z;
}

void cudaWarpAffine(uint8_t* input, uint32_t inputWidth, uint32_t inputHeight,
                    float* output, uint32_t outputWidth, uint32_t outputHeight,
                    float3 matrix[2], cudaStream_t stream) {
    // launch kernel
    const dim3 blockDim(8, 8);
    const dim3 gridDim(iDivUp(outputWidth, blockDim.x), iDivUp(outputHeight, blockDim.y));
    gpuBilinearWarpAffine<<<gridDim, blockDim, 0, stream>>>(input, inputWidth, inputHeight, output, outputWidth, outputHeight, matrix[0], matrix[1]);
}

}  // namespace deploy
