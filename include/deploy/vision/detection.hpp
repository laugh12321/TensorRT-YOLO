#pragma once

#include <memory>
#include <string>
#include <vector>

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"
#include "deploy/core/tensor.hpp"
#include "deploy/vision/cudaWarp.hpp"
#include "deploy/vision/result.hpp"

namespace deploy {

/**
 * @brief DeployDet class for performing object detection using YOLO series models.
 *
 * This class provides functionality for object detection using YOLO series models.
 * It supports both single image inference and batch inference.
 * CUDA acceleration can be optionally enabled for inference by setting the 'cudaMem'
 * flag to true during construction.
 */
class DEPLOY_DECL DeployDet {
public:
    // Constructor to initialize DeployDet with a model file and optional CUDA memory flag.
    explicit DeployDet(const std::string& file, bool cudaMem = false, int device = 0);

    // Destructor to clean up resources.
    ~DeployDet();

    // Perform object detection on a single image.
    DetectionResult predict(const Image& image);

    // Perform object detection on a batch of images.
    std::vector<DetectionResult> predict(const std::vector<Image>& images);

    // Batch size for inference.
    int batch{};

private:
    // Flag indicating whether CUDA memory is used.
    bool cudaMem{false};

    // Flag indicating dynamic allocation.
    bool dynamic{false};

    // Width and height of input images.
    int width{0}, height{0};

    // Engine context for inference.
    std::shared_ptr<EngineContext> engineCtx{};

    // Transformation matrices for preprocessing.
    std::vector<TransformMatrix> transforms{};

    // Information about input and output tensors.
    std::vector<TensorInfo> tensorInfos{};

    // Input tensors containing preprocessed images.
    std::vector<Tensor> imageTensors{};

    // CUDA streams for parallel execution.
    std::vector<cudaStream_t> inputStreams{};

    // CUDA stream for inference.
    cudaStream_t inferStream{nullptr};

    // Setup input and output tensors.
    void setupTensors();

    // Allocate memory for input and output tensors.
    void allocate();

    // Release allocated resources.
    void release();

    // Preprocess image before inference.
    void preProcess(int idx, const Image& image, cudaStream_t stream);

    // Post-process inference results.
    DetectionResult postProcess(int idx);
};

}  // namespace deploy