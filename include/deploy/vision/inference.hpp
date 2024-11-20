#pragma once

#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "deploy/core/core.hpp"
#include "deploy/core/macro.hpp"
#include "deploy/core/tensor.hpp"
#include "deploy/vision/cudaWarp.hpp"
#include "deploy/vision/result.hpp"

namespace deploy {

/**
 * @brief BaseTemplate class, a template base class for YOLO series models (e.g., Det, OBB, Seg, etc.).
 *
 * This class serves as a template base class providing common functionality for YOLO series models
 * (such as object detection, oriented bounding boxes, segmentation, etc.).
 * It supports both single image inference and batch inference.
 */
template <typename T>
class DEPLOYAPI BaseTemplate {
public:
    // Use static_assert to ensure that T is either DetResult, OBBResult, SegResult, or PoseResult..
    static_assert(
        std::is_same<T, DetResult>::value ||
            std::is_same<T, OBBResult>::value ||
            std::is_same<T, SegResult>::value ||
            std::is_same<T, PoseResult>::value,
        "T must be either DetResult, OBBResult, SegResult, or PoseResult.");

    /**
     * @brief Constructor to initialize BaseTemplate with a model file, optional CUDA memory flag, and device index.
     *
     * @param file The path to the model file.
     * @param cudaMem (Optional) Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
     * @param device (Optional) Device index for the inference. Defaults to 0.
     */
    explicit BaseTemplate(const std::string& file, bool cudaMem = false, int device = 0);

    /**
     * @brief Destructor for releasing allocated resources used by the BaseTemplate class.
     */
    virtual ~BaseTemplate() {};

    /**
     * @brief Performs inference on a single input image.
     *
     * @param image Input image for inference.
     * @return T Result of inference for the single image.
     */
    virtual T predict(const Image& image) = 0;

    /**
     * @brief Performs inference on a batch of input images.
     *
     * @param images Vector containing batch of input images for inference.
     * @return std::vector<T> Vector of inference results for each image in the batch.
     */
    virtual std::vector<T> predict(const std::vector<Image>& images) = 0;

    /**
     * @brief Sets the batch size for model inference.
     */
    int batch{};

protected:
    /**
     * @brief Flag indicating whether the input image is in GPU memory (true) or CPU memory (false).
     */
    bool cudaMem{false};

    /**
     * @brief Width and height of the input images used for inference.
     */
    int width{0}, height{0};

    /**
     * @brief Shared pointer to the engine context used for executing the inference model.
     */
    std::shared_ptr<EngineContext> engineCtx{};

    /**
     * @brief Transformation matrices applied to images during preprocessing.
     */
    std::vector<TransformMatrix> transforms{};

    /**
     * @brief Tensor information detailing the input and output tensors used by the model.
     */
    std::vector<TensorInfo> tensorInfos{};

    /**
     * @brief CUDA streams to handle input data transfer asynchronously.
     */
    std::vector<cudaStream_t> inputStreams{};

    /**
     * @brief CUDA stream used to execute inference operations.
     */
    cudaStream_t inferStream{nullptr};

    /**
     * @brief Allocates required resources for inference execution.
     */
    virtual void allocate() = 0;

    /**
     * @brief Releases resources that were allocated for inference.
     */
    virtual void release() = 0;

    /**
     * @brief Configures input and output tensors for model inference.
     */
    virtual void setupTensors() = 0;

    /**
     * @brief Processes the inference results for a specific index.
     *
     * @param idx Index corresponding to the result to be post-processed.
     * @return T Processed result of the specified inference output.
     */
    virtual T postProcess(int idx);
};

/**
 * @brief DeployTemplate class, a template class for YOLO series models (e.g., Det, OBB, Seg, etc.).
 *
 * This class is a specialized template class derived from BaseTemplate, providing functionality for
 * YOLO series models in tasks like object detection, oriented bounding boxes, segmentation, etc.
 * It supports both single image inference and batch inference, allowing for efficient deployment on
 * both GPU and CPU.
 */
template <typename T>
class DEPLOYAPI DeployTemplate : public BaseTemplate<T> {
public:
    // Use static_assert to ensure that T is either DetResult, OBBResult, SegResult, or PoseResult..
    static_assert(
        std::is_same<T, DetResult>::value ||
            std::is_same<T, OBBResult>::value ||
            std::is_same<T, SegResult>::value ||
            std::is_same<T, PoseResult>::value,
        "T must be either DetResult, OBBResult, SegResult, or PoseResult.");

    /**
     * @brief Constructor to initialize DeployTemplate with a model file, optional CUDA memory flag, and device index.
     *
     * @param file The path to the model file.
     * @param cudaMem (Optional) Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
     * @param device (Optional) Device index for the inference. Defaults to 0.
     */
    explicit DeployTemplate(const std::string& file, bool cudaMem = false, int device = 0);

    /**
     * @brief Destructor for releasing allocated resources used by the DeployTemplate class.
     */
    ~DeployTemplate();

    /**
     * @brief Performs inference on a single input image.
     *
     * @param image Input image for inference.
     * @return T Result of inference for the single image.
     */
    T predict(const Image& image) override;

    /**
     * @brief Performs inference on a batch of input images.
     *
     * @param images Vector containing batch of input images for inference.
     * @return std::vector<T> Vector of inference results for each image in the batch.
     */
    std::vector<T> predict(const std::vector<Image>& images) override;

private:
    /**
     * @brief Flag indicating whether the model is dynamic.
     */
    bool dynamic{false};

    /**
     * @brief Vector of input tensors for holding preprocessed image data.
     */
    std::vector<Tensor> imageTensors{};

    /**
     * @brief Allocates required resources for inference execution.
     */
    void allocate() override;

    /**
     * @brief Releases resources that were allocated for inference.
     */
    void release() override;

    /**
     * @brief Configures input and output tensors for model inference.
     */
    void setupTensors() override;

    /**
     * @brief Preprocesses a single image in the batch before inference.
     *
     * @param idx Index of the image in the batch.
     * @param image The input image to preprocess.
     * @param stream CUDA stream used to perform asynchronous operations for preprocessing.
     */
    void preProcess(int idx, const Image& image, cudaStream_t stream);
};

/**
 * @brief DeployCGTemplate class, a template class for YOLO series models (e.g., Det, OBB, Seg, etc.) using CUDA Graphs.
 *
 * This class is a specialized template derived from BaseTemplate, designed for YOLO series models
 * in tasks such as object detection, oriented bounding boxes, segmentation, etc.
 * DeployCGTemplate integrates CUDA Graphs to enhance inference efficiency and supports both single image and batch inference.
 * Only static models are supported; dynamic models are not supported.
 */
template <typename T>
class DEPLOYAPI DeployCGTemplate : public BaseTemplate<T> {
public:
    // Use static_assert to ensure that T is either DetResult, OBBResult, SegResult, or PoseResult..
    static_assert(
        std::is_same<T, DetResult>::value ||
            std::is_same<T, OBBResult>::value ||
            std::is_same<T, SegResult>::value ||
            std::is_same<T, PoseResult>::value,
        "T must be either DetResult, OBBResult, SegResult, or PoseResult.");

    /**
     * @brief Constructor to initialize DeployCGTemplate with a model file, optional CUDA memory flag, and device index.
     *
     * @param file The path to the model file.
     * @param cudaMem (Optional) Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
     * @param device (Optional) Device index for the inference. Defaults to 0.
     */
    explicit DeployCGTemplate(const std::string& file, bool cudaMem = false, int device = 0);

    /**
     * @brief Destructor for releasing allocated resources used by the DeployCGTemplate class.
     */
    ~DeployCGTemplate();

    /**
     * @brief Performs inference on a single input image.
     *
     * @param image Input image for inference.
     * @return T Result of inference for the single image.
     */
    T predict(const Image& image) override;

    /**
     * @brief Performs inference on a batch of input images.
     *
     * @param images Vector containing batch of input images for inference.
     * @return std::vector<T> Vector of inference results for each image in the batch.
     */
    std::vector<T> predict(const std::vector<Image>& images) override;

private:
    /**
     * @brief Number of elements in the input image.
     */
    int64_t inputSize{0};

    /**
     * @brief CUDA graph used for executing the inference workflow.
     */
    cudaGraph_t inferGraph{};

    /**
     * @brief Executable instance of the CUDA graph, created from inferGraph.
     */
    cudaGraphExec_t inferGraphExec{};

    /**
     * @brief Array of nodes in the CUDA graph.
     */
    std::unique_ptr<cudaGraphNode_t[]> graphNodes{};

    /**
     * @brief Parameters for CUDA kernel node in the graph.
     */
    std::vector<cudaKernelNodeParams> kernelsParams{};

    /**
     * @brief Parameters for memory copy operations within the graph.
     */
    cudaMemcpy3DParms memcpyParams;

    /**
     * @brief CUDA events used to synchronize input operations.
     */
    std::vector<cudaEvent_t> inputEvents{};

    /**
     * @brief Tensor for storing batched input images.
     */
    std::shared_ptr<Tensor> imageTensor{};

    /**
     * @brief Stores the number of elements for each input image in the batch.
     */
    std::vector<int64_t> imageSize{0};

    /**
     * @brief Allocates required resources for inference execution.
     */
    void allocate() override;

    /**
     * @brief Releases resources that were allocated for inference.
     */
    void release() override;

    /**
     * @brief Configures input and output tensors for model inference.
     */
    void setupTensors() override;

    /**
     * @brief Creates the CUDA graph for inference execution.
     */
    void createGraph();

    /**
     * @brief Retrieves and stores nodes in the CUDA graph.
     */
    void getGraphNodes();
};

// Explicitly instantiate the template class
template class BaseTemplate<DetResult>;
template class DeployTemplate<DetResult>;
template class DeployCGTemplate<DetResult>;
template class BaseTemplate<OBBResult>;
template class DeployTemplate<OBBResult>;
template class DeployCGTemplate<OBBResult>;
template class BaseTemplate<SegResult>;
template class DeployTemplate<SegResult>;
template class DeployCGTemplate<SegResult>;
template class BaseTemplate<PoseResult>;
template class DeployTemplate<PoseResult>;
template class DeployCGTemplate<PoseResult>;

// Use the template class to create concrete deployment classes
typedef BaseTemplate<DetResult>      BaseDet;
typedef DeployTemplate<DetResult>    DeployDet;
typedef DeployCGTemplate<DetResult>  DeployCGDet;
typedef BaseTemplate<OBBResult>      BaseOBB;
typedef DeployTemplate<OBBResult>    DeployOBB;
typedef DeployCGTemplate<OBBResult>  DeployCGOBB;
typedef BaseTemplate<SegResult>      BaseSeg;
typedef DeployTemplate<SegResult>    DeploySeg;
typedef DeployCGTemplate<SegResult>  DeployCGSeg;
typedef BaseTemplate<PoseResult>     BasePose;
typedef DeployTemplate<PoseResult>   DeployPose;
typedef DeployCGTemplate<PoseResult> DeployCGPose;

}  // namespace deploy
