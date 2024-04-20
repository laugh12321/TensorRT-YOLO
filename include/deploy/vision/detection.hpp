#pragma once

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"
#include "deploy/core/tensor.hpp"
#include "deploy/vision/result.hpp"
#include "deploy/vision/warp_affine.hpp"

namespace deploy {

class DEPLOY_DECL DeployDet {
public:
    DeployDet(const DeployDet&)            = default;
    DeployDet(DeployDet&&)                 = delete;
    DeployDet& operator=(const DeployDet&) = default;
    DeployDet& operator=(DeployDet&&)      = delete;
    /**
     * @brief Constructs a DeployDet object from a file.
     *
     * @param file Path to the serialized model file.
     * @throw std::runtime_error if file is empty or fails to construct engine
     * context.
     */
    explicit DeployDet(const std::string& file);

    /**
     * @brief Destructor.
     */
    ~DeployDet();

    /**
     * @brief Predicts the result for a single input image.
     *
     * @param image Input image.
     * @return Result Prediction result.
     */
    DetectionResult Predict(const cv::Mat& image);

    /**
     * @brief Predicts the results for multiple input images.
     *
     * @param images Vector of input images.
     * @return std::vector<Result> Vector of prediction results.
     */
    std::vector<DetectionResult> Predict(const std::vector<cv::Mat>& images);

    int batch_size{};

private:
    int                            width_{}, height_{};
    std::vector<TensorInfo>        tensors_info_{};
    std::shared_ptr<EngineContext> engine_ctx_{};
    std::vector<Tensor>            input_tensors_{};
    std::vector<cudaStream_t>      input_streams_{};
    std::vector<AffineTransform>   input_transforms_{};
    cudaStream_t                   infer_stream_{};

    void Setup();

    /**
     * @brief Allocates resources required for inference.
     */
    void Allocate();

    /**
     * @brief Releases allocated resources.
     */
    void Release();

    /**
     * @brief Executes inference on input tensors.
     */
    bool Infer();

    /**
     * @brief Preprocesses the input image.
     *
     * @param idx Index of the input image.
     * @param image Input image.
     * @param stream CUDA stream for asynchronous processing.
     */
    void PreProcess(int idx, const cv::Mat& image, cudaStream_t stream);

    /**
     * @brief Postprocesses the inference result.
     *
     * @param idx Index of the inference result.
     * @return Result Processed prediction result.
     */
    DetectionResult PostProcess(int idx);
};

}  // namespace deploy
