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
 * @brief BaseDet class for common functionality of object detection using YOLO series models.
 *
 * This class provides common functionality for object detection using YOLO series models.
 * It supports both single image inference and batch inference.
 * CUDA acceleration can be optionally enabled for inference by setting the 'cudaMem'
 * flag to true during construction. The class also supports detection with oriented bounding boxes (OBB).
 */
class DEPLOY_DECL BaseDet {
public:
    /**
     * @brief Constructor to initialize BaseDet with a model file, OBB flag, optional CUDA memory flag, and device index.
     *
     * @param file The path to the model file.
     * @param obb Flag indicating whether the model uses oriented bounding boxes (OBB). If true, the model is expected
     *            to handle rotated bounding boxes.
     * @param cudaMem (Optional) Flag to enable CUDA memory usage for inference. Defaults to false.
     * @param device (Optional) Device index to use for inference. Defaults to 0.
     */
    explicit BaseDet(const std::string& file, bool obb, bool cudaMem = false, int device = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    virtual ~BaseDet(){};

    /**
     * @brief Perform object detection on a single image.
     *
     * @param image The input image on which to perform object detection.
     * @return DetectionResult The detection results for the input image.
     */
    virtual DetectionResult predict(const Image& image) = 0;

    /**
     * @brief Perform object detection on a batch of images.
     *
     * @param images A vector of images on which to perform object detection.
     * @return std::vector<DetectionResult> The detection results for the batch of images.
     */
    virtual std::vector<DetectionResult> predict(const std::vector<Image>& images) = 0;

    /**
     * @brief Batch size for inference.
     */
    int batch{};

protected:
    /**
     * @brief Flag indicating whether the model uses oriented bounding boxes (OBB).
     */
    bool obb{false};

    /**
     * @brief Flag indicating whether CUDA memory is used.
     */
    bool cudaMem{false};

    /**
     * @brief Width and height of input images.
     */
    int width{0}, height{0};

    /**
     * @brief Engine context for inference.
     */
    std::shared_ptr<EngineContext> engineCtx{};

    /**
     * @brief Transformation matrices for preprocessing.
     */
    std::vector<TransformMatrix> transforms{};

    /**
     * @brief Information about input and output tensors.
     */
    std::vector<TensorInfo> tensorInfos{};

    /**
     * @brief CUDA streams for parallel execution.
     */
    std::vector<cudaStream_t> inputStreams{};

    /**
     * @brief CUDA stream for inference.
     */
    cudaStream_t inferStream{nullptr};

    /**
     * @brief Allocate necessary resources.
     */
    virtual void allocate() = 0;

    /**
     * @brief Release allocated resources.
     */
    virtual void release() = 0;

    /**
     * @brief Setup input and output tensors.
     */
    virtual void setupTensors() = 0;

    /**
     * @brief Post-process inference results.
     *
     * @param idx Index of the result to process.
     * @return DetectionResult Processed detection result.
     */
    virtual DetectionResult postProcess(int idx);
};

/**
 * @brief DeployDet class for performing object detection using YOLO series models.
 *
 * This class provides functionality for object detection using YOLO series models.
 * It supports both single image inference and batch inference.
 * CUDA acceleration can be optionally enabled for inference by setting the 'cudaMem'
 * flag to true during construction.
 */
class DEPLOY_DECL DeployDet : public BaseDet {
public:
    /**
     * @brief Constructor to initialize DeployDet with a model file, OBB flag, optional CUDA memory flag, and device index.
     *
     * @param file The path to the model file.
     * @param obb Flag indicating whether the model uses oriented bounding boxes (OBB).
     * @param cudaMem (Optional) Flag to enable CUDA memory usage for inference. Defaults to false.
     * @param device (Optional) Device index to use for inference. Defaults to 0.
     */
    explicit DeployDet(const std::string& file, bool obb, bool cudaMem = false, int device = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    ~DeployDet();

    /**
     * @brief Perform object detection on a single image.
     *
     * @param image The input image on which to perform object detection.
     * @return DetectionResult The detection results for the input image.
     */
    DetectionResult predict(const Image& image) override;

    /**
     * @brief Perform object detection on a batch of images.
     *
     * @param images A vector of images on which to perform object detection.
     * @return std::vector<DetectionResult> The detection results for the batch of images.
     */
    std::vector<DetectionResult> predict(const std::vector<Image>& images) override;

private:
    /**
     * @brief Flag indicating dynamic allocation.
     */
    bool dynamic{false};

    /**
     * @brief Input tensors containing preprocessed images.
     */
    std::vector<Tensor> imageTensors{};

    /**
     * @brief Allocate necessary resources.
     */
    void allocate() override;

    /**
     * @brief Release allocated resources.
     */
    void release() override;

    /**
     * @brief Setup input and output tensors.
     */
    void setupTensors() override;

    /**
     * @brief Preprocess image before inference.
     *
     * @param idx Index of the image in the batch.
     * @param image The input image to preprocess.
     * @param stream The CUDA stream to use for preprocessing.
     */
    void preProcess(int idx, const Image& image, cudaStream_t stream);
};

/**
 * @brief DeployCGDet class for performing object detection with CUDA Graphs using YOLO series models.
 *
 * This class extends the DeployDet class and provides additional functionality using CUDA Graphs
 * for efficient inference.
 */
class DEPLOY_DECL DeployCGDet : public BaseDet {
public:
    /**
     * @brief Constructor to initialize DeployCGDet with a model file, OBB flag, optional CUDA memory flag, and device index.
     *
     * @param file The path to the model file.
     * @param obb Flag indicating whether the model uses oriented bounding boxes (OBB).
     * @param cudaMem (Optional) Flag to enable CUDA memory usage for inference. Defaults to false.
     * @param device (Optional) Device index to use for inference. Defaults to 0.
     */
    explicit DeployCGDet(const std::string& file, bool obb, bool cudaMem = false, int device = 0);

    /**
     * @brief Destructor to clean up resources.
     */
    ~DeployCGDet();

    /**
     * @brief Perform object detection on a single image.
     *
     * @param image The input image on which to perform object detection.
     * @return DetectionResult The detection results for the input image.
     */
    DetectionResult predict(const Image& image) override;

    /**
     * @brief Perform object detection on a batch of images.
     *
     * @param images A vector of images on which to perform object detection.
     * @return std::vector<DetectionResult> The detection results for the batch of images.
     */
    std::vector<DetectionResult> predict(const std::vector<Image>& images) override;

private:
    /**
     * @brief Input image size.
     */
    int inputSize{0};

    /**
     * @brief CUDA graph and its executable instance.
     */
    cudaGraph_t inferGraph{};

    /**
     * @brief Executable instance of the CUDA graph.
     */
    cudaGraphExec_t inferGraphExec{};

    /**
     * @brief Nodes in the CUDA graph.
     */
    std::unique_ptr<cudaGraphNode_t[]> graphNodes{};

    /**
     * @brief Parameters for CUDA kernel nodes.
     */
    std::vector<cudaKernelNodeParams> kernelsParams{};

    /**
     * @brief Parameters for CUDA memory copy operations.
     */
    cudaMemcpy3DParms memcpyParams;

    /**
     * @brief CUDA events for synchronizing input operations.
     */
    std::vector<cudaEvent_t> inputEvents{};

    /**
     * @brief Tensor for storing input images.
     */
    std::shared_ptr<Tensor> imageTensor{};

    /**
     * @brief Sizes of the input images.
     */
    std::vector<int> imageSize{0};

    /**
     * @brief Allocate necessary resources.
     */
    void allocate() override;

    /**
     * @brief Release allocated resources.
     */
    void release() override;

    /**
     * @brief Setup input and output tensors.
     */
    void setupTensors() override;

    /**
     * @brief Create CUDA graph for inference.
     */
    void createGraph();

    /**
     * @brief Retrieve graph nodes for the CUDA graph.
     */
    void getGraphNodes();
};

}  // namespace deploy
