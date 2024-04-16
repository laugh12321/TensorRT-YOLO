#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

#include "common.hpp"
#include "cuda_utils.hpp"

namespace yolo {

// Definition of TensorInfo structure
struct TensorInfo {
    int32_t                                    index{-1};
    char const*                                name{nullptr};
    nvinfer1::Dims                             dims{};
    bool                                       isDynamic{};
    bool                                       isInput{};
    nvinfer1::DataType                         dataType{};
    int64_t                                    vol{-1};
    std::shared_ptr<cuda_utils::HostDeviceMem> memory;

    // Update volume of tensor
    void updateVolume() { vol = cuda_utils::volume(dims); }
};

class Detection {
   public:
    ~Detection();
    bool       load(const std::string& filePath);
    DetectInfo predict(const cv::Mat& image, cudaStream_t stream = 0);
    std::vector<DetectInfo> predict(const std::vector<cv::Mat>& images,
                                    cudaStream_t                stream = 0);
    int                     batchSize;

   private:
    int                                        _inputWidth, _inputHeight;
    std::vector<TensorInfo>                    _tensorInfos;
    std::shared_ptr<cuda_utils::EngineContext> _engineCtx;
    std::vector<cudaStream_t>                  _streams;
    std::vector<cuda_utils::HostDeviceMem>     _preBuffers;
    std::vector<AffineTransform>               _affineTransforms;
    // Initialization
    void                                       setup();
    // Release resources
    void                                       release();
    // Allocate memory
    void                                       allocate();
    // Inference
    bool                                       infer(cudaStream_t stream);
    // Create tensor information
    TensorInfo                                 createTensorInfo(int index);
    // Preprocessing
    void preprocess(const int idx, const cv::Mat& image, cudaStream_t stream);
    // Postprocessing
    DetectInfo postprocess(const int idx);
};

// Load model
std::shared_ptr<Detection> load(const std::string& filePath);

}  // namespace yolo
