#pragma once

#include <NvInferRuntime.h>

#include <cstddef>
#include <cstdint>

namespace deploy {

constexpr int defaultAlignment = 32;

/**
 * @brief Gets the size in bytes of a given data type.
 *
 * @param dataType The data type.
 * @return size_t The size in bytes of the data type.
 */
size_t getDataTypeSize(nvinfer1::DataType dataType);

/**
 * @brief Calculates the volume (total number of elements) of a tensor described
 * by dims.
 *
 * @param dims The dimensions of the tensor.
 * @return int64_t The total number of elements in the tensor.
 */
int64_t calculateVolume(nvinfer1::Dims const& dims);

/**
 * @brief Rounds up n to the nearest multiple of align.
 *
 * @param n The number to be rounded up.
 * @param align The alignment value (default is defaultAlignment).
 * @return int64_t The rounded up value.
 */
int64_t roundUp(int64_t n, int64_t align = defaultAlignment);

}  // namespace deploy
