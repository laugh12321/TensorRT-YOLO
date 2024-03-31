#pragma once

#include "common.hpp"
#include "cuda_utils.hpp"
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace yolo {

struct TensorInfo {
    int32_t index{-1};
    char const* name{nullptr};
    nvinfer1::Dims dims{};
    bool isDynamic{};
    bool isInput{};
    nvinfer1::DataType dataType{};
    int64_t vol{-1};
    std::shared_ptr<cuda_utils::HostDeviceMem> memory;

    void updateVolume() {
        vol = cuda_utils::volume(dims);
    }
};

class Detection {
public:
    int batchSize;
    bool load(const std::string& filePath);
    DetectInfo predict(const cv::Mat& image, cudaStream_t stream = 0);
    std::vector<DetectInfo> predict(const std::vector<cv::Mat>& images, cudaStream_t stream = 0);

private:
    int _inputWidth, _inputHeight;
    std::vector<TensorInfo> _tensorInfos;
    std::vector<AffineTransform> _affineTransforms;
    std::shared_ptr<cuda_utils::EngineContext> _engineCtx;
    std::vector<std::shared_ptr<cuda_utils::HostDeviceMem>> _preBuffers;

    void setup();
    bool infer(cudaStream_t stream);
    TensorInfo createTensorInfo(int index);
    void preprocess(const int idx, const cv::Mat& image, cudaStream_t stream);
    DetectInfo postprocess(const int idx);
};

std::shared_ptr<Detection> load(const std::string& filePath);

} // namespace yolo
