#pragma once

#include <NvInferRuntimeBase.h>

#include <cstddef>
#include <cstdint>

namespace deploy {

constexpr int DefaultAlignment = 32;

/**
 * @brief Gets the size in bytes of a given data type.
 *
 * @param data_type The data type.
 * @return size_t The size in bytes of the data type.
 */
size_t GetDataTypeSize(nvinfer1::DataType data_type);

/**
 * @brief Calculates the volume (total number of elements) of a tensor described
 * by dims.
 *
 * @param dims The dimensions of the tensor.
 * @return int64_t The total number of elements in the tensor.
 */
int64_t CalculateVolume(nvinfer1::Dims const& dims);

/**
 * @brief Rounds up n to the nearest multiple of align.
 *
 * @param n The number to be rounded up.
 * @param align The alignment value (default is DefaultAlignment).
 * @return int64_t The rounded up value.
 */
int64_t RoundUp(int64_t n, int64_t align = DefaultAlignment);

}  // namespace deploy
