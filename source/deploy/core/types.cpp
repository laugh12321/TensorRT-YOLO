#include "deploy/core/types.hpp"

namespace deploy {

size_t getDataTypeSize(nvinfer1::DataType dataType) {
    switch (dataType) {
        case nvinfer1::DataType::kINT32:
        case nvinfer1::DataType::kFLOAT:
            return 4U;
        case nvinfer1::DataType::kHALF:
            return 2U;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8:
            return 1U;
    }
    return 0;
}

int64_t calculateVolume(const nvinfer1::Dims& dims) {
    int64_t volume = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        volume *= static_cast<int64_t>(dims.d[i]);
    }
    return volume;
}

int64_t roundUp(int64_t n, int64_t align) {
    return (n + align - 1) / align * align;
}

}  // namespace deploy