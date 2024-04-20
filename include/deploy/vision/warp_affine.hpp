#pragma once

#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

namespace deploy {

/**
 * @brief Represents an affine transformation manager.
 */
struct AffineTransform {
    float    warp_matrix[6]; /**< Warp affine transformation matrix */
    cv::Size last_size;      /**< Last size used for transformation */

    /**
     * @brief Calculates the warp affine transformation matrix.
     *
     * @param from_size Size of the source image.
     * @param to_size Size of the target image.
     */
    void CalculateWarpMatrix(const cv::Size& from_size,
                             const cv::Size& to_size);

    /**
     * @brief Applies the warp affine transformation matrix to a point.
     *
     * @param x X-coordinate of the point.
     * @param y Y-coordinate of the point.
     * @param[out] ox Transformed X-coordinate.
     * @param[out] oy Transformed Y-coordinate.
     */
    void ApplyWarp(float x, float y, float* ox, float* oy) const;
};

/**
 * @brief Performs bilinear warp affine transformation on an image.
 *
 * @param src Pointer to the source image data.
 * @param src_line_size Length of each line in the source image.
 * @param src_width Width of the source image.
 * @param src_height Height of the source image.
 * @param dst Pointer to the destination image data.
 * @param dst_width Width of the destination image.
 * @param dst_height Height of the destination image.
 * @param matrix Pointer to the warp affine transformation matrix.
 * @param const_value Constant value to use for out-of-range pixels.
 * @param stream CUDA stream for asynchronous execution.
 */
void BilinearWarpAffine(uint8_t* src, int src_line_size, int src_width,
                        int src_height, float* dst, int dst_width,
                        int dst_height, float* matrix, uint8_t const_value,
                        cudaStream_t stream);

}  // namespace deploy
