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
 * flag to true during construction.
 */
class DEPLOY_DECL BaseDet {
public:
    // Constructor to initialize BaseDet with a model file and optional CUDA memory flag.
    explicit BaseDet(const std::string& file, bool cudaMem = false, int device = 0);

    // Destructor to clean up resources.
    virtual ~BaseDet(){};

    // Perform object detection on a single image.
    virtual DetectionResult predict(const Image& image) = 0;

    // Perform object detection on a batch of images.
    virtual std::vector<DetectionResult> predict(const std::vector<Image>& images) = 0;

    // Batch size for inference.
    int batch{};

protected:
    // Flag indicating whether CUDA memory is used.
    bool                           cudaMem{false};
    // Width and height of input images.
    int                            width{0}, height{0};
    // Engine context for inference.
    std::shared_ptr<EngineContext> engineCtx{};
    // Transformation matrices for preprocessing.
    std::vector<TransformMatrix>   transforms{};
    // Information about input and output tensors.
    std::vector<TensorInfo>        tensorInfos{};
    // CUDA streams for parallel execution.
    std::vector<cudaStream_t>      inputStreams{};
    // CUDA stream for inference.
    cudaStream_t                   inferStream{nullptr};
    // Allocate necessary resources.
    virtual void                   allocate()     = 0;
    // Release allocated resources.
    virtual void                   release()      = 0;
    // Setup input and output tensors.
    virtual void                   setupTensors() = 0;
    // Post-process inference results.
    virtual DetectionResult        postProcess(int idx);
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
    // Constructor to initialize DeployDet with a model file and optional CUDA memory flag.
    explicit DeployDet(const std::string& file, bool cudaMem = false, int device = 0);

    // Destructor to clean up resources.
    ~DeployDet();

    // Perform object detection on a single image.
    DetectionResult predict(const Image& image) override;

    // Perform object detection on a batch of images.
    std::vector<DetectionResult> predict(const std::vector<Image>& images) override;

private:
    // Flag indicating dynamic allocation.
    bool                dynamic{false};
    // Input tensors containing preprocessed images.
    std::vector<Tensor> imageTensors{};
    // Allocate necessary resources.
    void                allocate() override;
    // Release allocated resources.
    void                release() override;
    // Setup input and output tensors.
    void                setupTensors() override;
    // Preprocess image before inference.
    void                preProcess(int idx, const Image& image, cudaStream_t stream);
};

class DEPLOY_DECL DeployCGDet : public BaseDet {
public:
    // Constructor to initialize DeployCGDet with a model file and optional CUDA memory flag.
    explicit DeployCGDet(const std::string& file, bool cudaMem = false, int device = 0);

    // Destructor to clean up resources.
    ~DeployCGDet();

    // Perform object detection on a single image.
    DetectionResult predict(const Image& image) override;

    // Perform object detection on a batch of images.
    std::vector<DetectionResult> predict(const std::vector<Image>& images) override;

private:
    // input images.
    int                                inputSize{0};
    // CUDA graph and its executable instance.
    cudaGraph_t                        inferGraph{};
    cudaGraphExec_t                    inferGraphExec{};
    // Nodes in the CUDA graph.
    std::unique_ptr<cudaGraphNode_t[]> graphNodes{};
    // Parameters for CUDA kernel nodes.
    std::vector<cudaKernelNodeParams>  kernelsParams{};
    // Parameters for CUDA memory copy operations.
    cudaMemcpy3DParms                  memcpyParams;
    // CUDA events for synchronizing input operations.
    std::vector<cudaEvent_t>           inputEvents{};
    // Tensor for storing input images.
    std::shared_ptr<Tensor>            imageTensor{};
    // Sizes of the input images.
    std::vector<int>                   imageSize{0};
    // Allocate necessary resources.
    void                               allocate() override;
    // Release allocated resources.
    void                               release() override;
    // Setup input and output tensors.
    void                               setupTensors() override;
    // Create CUDA graph for inference.
    void                               createGraph();
    // Retrieve graph nodes for the CUDA graph.
    void                               getGraphNodes();
};

}  // namespace deploy