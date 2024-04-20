#include <algorithm>

#include "deploy/vision/warp_affine.hpp"

namespace deploy {

void AffineTransform::CalculateWarpMatrix(const cv::Size &from_size,
                                          const cv::Size &to_size) {
    if (from_size == last_size) return;
    last_size = from_size;

    double scale =
        std::min(static_cast<double>(to_size.width) / from_size.width,
                 static_cast<double>(to_size.height) / from_size.height);
    double offset = 0.5 * scale - 0.5;

    double scale_from_width  = -0.5 * scale * from_size.width;
    double scale_from_height = -0.5 * scale * from_size.height;
    double half_to_width     = 0.5 * to_size.width;
    double half_to_height    = 0.5 * to_size.height;

    double inv_d = (scale != 0.0) ? 1.0 / (scale * scale) : 0.0;
    double A     = scale * inv_d;

    warp_matrix[0] = A;
    warp_matrix[1] = 0.0;
    warp_matrix[2] = -A * (scale_from_width + half_to_width + offset);
    warp_matrix[3] = 0.0;
    warp_matrix[4] = A;
    warp_matrix[5] = -A * (scale_from_height + half_to_height + offset);
}

void AffineTransform::ApplyWarp(float x, float y, float *ox, float *oy) const {
    *ox = static_cast<float>(warp_matrix[0] * x + warp_matrix[1] * y +
                             warp_matrix[2]);
    *oy = static_cast<float>(warp_matrix[3] * x + warp_matrix[4] * y +
                             warp_matrix[5]);
}

__device__ float BilinearInterpolation(uint8_t *src, int src_line_size,
                                       int src_width, int src_height,
                                       float src_x, float src_y, int channel) {
    int x_low  = floorf(src_x);
    int y_low  = floorf(src_y);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    // Clamp coordinates to image boundaries
    x_low  = max(0, min(x_low, src_width - 1));
    x_high = max(0, min(x_high, src_width - 1));
    y_low  = max(0, min(y_low, src_height - 1));
    y_high = max(0, min(y_high, src_height - 1));

    // Compute interpolation weights
    float lx = src_x - x_low;
    float ly = src_y - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;

    // Compute pixel pointers
    uint8_t *v1 = src + y_low * src_line_size + x_low * 3 + channel;
    uint8_t *v2 = src + y_low * src_line_size + x_high * 3 + channel;
    uint8_t *v3 = src + y_high * src_line_size + x_low * 3 + channel;
    uint8_t *v4 = src + y_high * src_line_size + x_high * 3 + channel;

    // Perform bilinear interpolation
    return hy * (hx * (*v1) + lx * (*v2)) + ly * (hx * (*v3) + lx * (*v4));
}

__global__ void BilinearWarpAffineKernel(uint8_t *src, int src_line_size,
                                         int src_width, int src_height,
                                         float *dst, int dst_width,
                                         int dst_height, float *matrix,
                                         uint8_t const_value) {
    int dx   = threadIdx.x + blockIdx.x * blockDim.x;
    int dy   = threadIdx.y + blockIdx.y * blockDim.y;
    int area = dst_width * dst_height;
    if (dx >= dst_width || dy >= dst_height) return;

    float src_x = matrix[0] * dx + matrix[1] * dy + matrix[2];
    float src_y = matrix[3] * dx + matrix[4] * dy + matrix[5];

    // If out of range, set to constant value
    float c0 = const_value, c1 = const_value, c2 = const_value;

    // Precompute interpolation coefficients
    if (src_x > -1 && src_x < src_width && src_y > -1 && src_y < src_height) {
        // Perform bilinear interpolation for each channel
        c0 = BilinearInterpolation(src, src_line_size, src_width, src_height,
                                   src_x, src_y, 0);
        c1 = BilinearInterpolation(src, src_line_size, src_width, src_height,
                                   src_x, src_y, 1);
        c2 = BilinearInterpolation(src, src_line_size, src_width, src_height,
                                   src_x, src_y, 2);
    }

    // Swap B and R channels
    float temp = c2;
    c2         = c0;
    c0         = temp;

    // Normalize values to range [0, 1]
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // Reorder RGB to RRRGGGBBB
    float *_p_dst_red   = dst + dy * dst_width + dx;
    float *_p_dst_green = _p_dst_red + area;
    float *_p_dst_blue  = _p_dst_green + area;

    *_p_dst_red   = c0;
    *_p_dst_green = c1;
    *_p_dst_blue  = c2;
}

void BilinearWarpAffine(uint8_t *src, int src_line_size, int src_width,
                        int src_height, float *dst, int dst_width,
                        int dst_height, float *matrix, uint8_t const_value, cudaStream_t stream) {
    dim3 blockSize(32, 32);
    dim3 gridSize((dst_width + blockSize.x - 1) / blockSize.x, (dst_height + blockSize.y - 1) / blockSize.y);

    BilinearWarpAffineKernel<<<gridSize, blockSize, 0, stream>>>(
        src, src_line_size, src_width, src_height,
        dst, dst_width, dst_height,
        matrix, const_value);
}

}  // namespace deploy
