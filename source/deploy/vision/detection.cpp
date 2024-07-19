#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "deploy/core/types.hpp"
#include "deploy/utils/utils.hpp"
#include "deploy/vision/detection.hpp"

namespace deploy {

// Constructor to initialize BaseDet with a model file and optional CUDA memory flag.
BaseDet::BaseDet(const std::string& file, bool cudaMem, int device) : cudaMem(cudaMem) {
    // Set the CUDA device
    CUDA(cudaSetDevice(device));

    // Load the engine data from file
    auto data = loadFile(file);

    // Reset engine context
    engineCtx = std::make_shared<EngineContext>();
    if (!engineCtx->construct(data.data(), data.size())) {
        throw std::runtime_error("Failed to construct engine context.");
    }
}

// Post-process inference results.
DetectionResult BaseDet::postProcess(const int idx) {
    int    num     = static_cast<int*>(tensorInfos[1].tensor.host())[idx];
    float* boxes   = static_cast<float*>(tensorInfos[2].tensor.host()) + idx * tensorInfos[2].dims.d[1] * tensorInfos[2].dims.d[2];
    float* scores  = static_cast<float*>(tensorInfos[3].tensor.host()) + idx * tensorInfos[3].dims.d[1];
    int*   classes = static_cast<int*>(tensorInfos[4].tensor.host()) + idx * tensorInfos[4].dims.d[1];

    DetectionResult result;
    result.num = num;

    for (int i = 0; i < num; ++i) {
        float left   = boxes[i * 4];
        float top    = boxes[i * 4 + 1];
        float right  = boxes[i * 4 + 2];
        float bottom = boxes[i * 4 + 3];

        // Apply affine transformation
        transforms[idx].transform(left, top, &left, &top);
        transforms[idx].transform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.emplace_back(scores[i]);
        result.classes.emplace_back(classes[i]);
    }

    return result;
}

// Constructor to initialize DeployDet with a model file and optional CUDA memory flag.
DeployDet::DeployDet(const std::string& file, bool cudaMem, int device) : BaseDet(file, cudaMem, device) {
    // Setup tensors based on the engine context
    setupTensors();

    // Allocate necessary resources
    allocate();
}

// Destructor to clean up resources.
DeployDet::~DeployDet() {
    release();
}

// Allocate necessary resources.
void DeployDet::allocate() {
    // Create infer stream
    CUDA(cudaStreamCreate(&inferStream));

    // Create input streams
    inputStreams.resize(batch);
    for (auto& stream : inputStreams) {
        CUDA(cudaStreamCreate(&stream));
    }

    // Allocate transforms and image tensors
    transforms.resize(batch, TransformMatrix());
    if (!cudaMem) imageTensors.resize(batch, Tensor());
}

// Release allocated resources.
void DeployDet::release() {
    // Release infer stream
    if (inferStream != nullptr) {
        CUDA(cudaStreamDestroy(inferStream));
        inferStream = nullptr;
    }

    // Release input streams
    for (auto& stream : inputStreams) {
        if (stream != nullptr) {
            CUDA(cudaStreamDestroy(stream));
        }
    }
    inputStreams.clear();

    // Release other resources
    transforms.clear();
    tensorInfos.clear();
    engineCtx.reset();
    if (!cudaMem) imageTensors.clear();
}

// Setup input and output tensors.
void DeployDet::setupTensors() {
    int tensorNum = engineCtx->mEngine->getNbIOTensors();
    tensorInfos.reserve(tensorNum);
    for (size_t i = 0; i < tensorNum; i++) {
        const char* name   = engineCtx->mEngine->getIOTensorName(i);
        auto        dims   = engineCtx->mEngine->getTensorShape(name);
        auto        dtype  = engineCtx->mEngine->getTensorDataType(name);
        bool        input  = (engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        size_t      typesz = getDataTypeSize(dtype);

        if (input) {
            dynamic = std::any_of(dims.d, dims.d + dims.nbDims, [](int val) { return val == -1; });
            if (dynamic) dims = engineCtx->mEngine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            batch  = dims.d[0];
            height = dims.d[2];
            width  = dims.d[3];
        } else if (!input && dynamic) {
            dims.d[0] = batch;
        }

        int64_t bytes = calculateVolume(dims) * typesz;
        tensorInfos.emplace_back(name, dims, input, typesz, bytes);
    }
}

// Preprocess image before inference.
void DeployDet::preProcess(const int idx, const Image& image, cudaStream_t stream) {
    transforms[idx].update(image.width, image.height, width, height);

    int64_t inputSize   = 3 * height * width;
    float*  inputDevice = static_cast<float*>(tensorInfos[0].tensor.device()) + idx * inputSize;

    void* imageDevice = nullptr;
    if (cudaMem) {
        imageDevice = image.rgbPtr;
    } else {
        int64_t imageSize = 3 * image.width * image.height;
        imageDevice       = imageTensors[idx].device(imageSize);
        void* imageHost   = imageTensors[idx].host(imageSize);

        std::memcpy(imageHost, image.rgbPtr, imageSize * sizeof(uint8_t));
        CUDA(cudaMemcpyAsync(imageDevice, imageHost, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    }

    cudaWarpAffine(static_cast<uint8_t*>(imageDevice), image.width, image.height, inputDevice, width, height, transforms[idx].matrix, stream);
}

// Perform object detection on a single image.
DetectionResult DeployDet::predict(const Image& image) {
    auto results = predict(std::vector<Image>{image});
    if (results.empty()) {
        return DetectionResult();
    }
    return results[0];
}

// Perform object detection on a batch of images.
std::vector<DetectionResult> DeployDet::predict(const std::vector<Image>& images) {
    std::vector<DetectionResult> results;
    int                          numImages = images.size();
    if (numImages < 1 || numImages > batch) {
        std::cerr << "Error: Number of images (" << numImages << ") must be between 1 and " << batch << " inclusive." << std::endl;
        return results;
    }

    for (auto& tensorInfo : tensorInfos) {
        tensorInfo.dims.d[0] = numImages;
        if (dynamic) tensorInfo.update();
        if (!tensorInfo.input) tensorInfo.tensor.host(tensorInfo.bytes);

        engineCtx->mContext->setTensorAddress(tensorInfo.name.data(), tensorInfo.tensor.device(tensorInfo.bytes));
        if (tensorInfo.input && dynamic) {
            engineCtx->mContext->setInputShape(tensorInfo.name.data(), tensorInfo.dims);
        }
    }

    if (numImages > 1) {
        for (size_t i = 0; i < numImages; ++i) {
            preProcess(i, images[i], inputStreams[i]);
        }

        for (auto& stream : inputStreams) {
            CUDA(cudaStreamSynchronize(stream));
        }
    } else {
        preProcess(0, images[0], inferStream);
    }

    if (!engineCtx->mContext->enqueueV3(inferStream)) return {};

    for (auto& tensorInfo : tensorInfos) {
        if (!tensorInfo.input) {
            CUDA(cudaMemcpyAsync(tensorInfo.tensor.host(), tensorInfo.tensor.device(), tensorInfo.bytes, cudaMemcpyDeviceToHost, inferStream));
        }
    }

    CUDA(cudaStreamSynchronize(inferStream));

    results.reserve(numImages);
    for (int i = 0; i < numImages; ++i) {
        results.emplace_back(postProcess(i));
    }

    return results;
}

// Constructor to initialize DeployCGDet with a model file and optional CUDA memory flag.
DeployCGDet::DeployCGDet(const std::string& file, bool cudaMem, int device) : BaseDet(file, cudaMem, device) {
    // Setup tensors based on the engine context
    setupTensors();

    // Allocate necessary resources
    allocate();

    // Create the CUDA graph
    createGraph();

    // Retrieve nodes from the CUDA graph
    getGraphNodes();

    // If CUDA memory optimization is enabled, reset the image tensor
    if (cudaMem) {
        imageTensor.reset();
    }
}

// Destructor to clean up resources.
DeployCGDet::~DeployCGDet() {
    release();
}

// Allocate necessary resources.
void DeployCGDet::allocate() {
    // Create the main inference stream
    CUDA(cudaStreamCreate(&inferStream));

    // Create resources for batched inputs
    if (batch > 1) {
        inputStreams.reserve(batch);
        for (int i = 0; i < batch; i++) {
            cudaStream_t stream;
            CUDA(cudaStreamCreate(&stream));
            inputStreams.push_back(stream);
        }

        inputEvents.reserve(batch * 2);
        for (int i = 0; i < batch * 2; i++) {
            cudaEvent_t event;
            CUDA(cudaEventCreate(&event));
            inputEvents.push_back(event);
        }
    }

    // Allocate memory for tensors
    kernelsParams.reserve(batch);
    imageSize.reserve(batch);
    transforms.reserve(batch);
    imageTensor = std::make_shared<Tensor>();

    inputSize = width * height * 3;
    imageTensor->device(inputSize * sizeof(uint8_t) * batch);
    imageTensor->host(inputSize * sizeof(uint8_t) * batch);

    // Update transform matrices for each batch
    for (size_t i = 0; i < batch; i++) {
        transforms[i].update(width, height, width, height);
    }
}

// Release allocated resources.
void DeployCGDet::release() {
    // Release CUDA graph execution
    if (inferGraphExec != nullptr) {
        CUDA(cudaGraphExecDestroy(inferGraphExec));
        inferGraphExec = nullptr;
    }

    // Release CUDA graph
    if (inferGraph != nullptr) {
        CUDA(cudaGraphDestroy(inferGraph));
        inferGraph = nullptr;
    }

    // Release CUDA stream
    if (inferStream != nullptr) {
        CUDA(cudaStreamDestroy(inferStream));
        inferStream = nullptr;
    }

    // Release resources for batched inputs
    if (batch > 1) {
        // Release input streams
        for (auto& stream : inputStreams) {
            if (stream != nullptr) {
                CUDA(cudaStreamDestroy(stream));
            }
        }
        inputStreams.clear();

        // Release input events
        for (auto& event : inputEvents) {
            if (event != nullptr) {
                CUDA(cudaEventDestroy(event));
            }
        }
        inputEvents.clear();
    }

    // Release other resources
    kernelsParams.clear();
    imageSize.clear();
    tensorInfos.clear();
    transforms.clear();
    engineCtx.reset();
    imageTensor.reset();
}

// Setup input and output tensors.
void DeployCGDet::setupTensors() {
    int tensorNum = engineCtx->mEngine->getNbIOTensors();
    tensorInfos.reserve(tensorNum);
    for (size_t i = 0; i < tensorNum; i++) {
        const char* name   = engineCtx->mEngine->getIOTensorName(i);
        auto        dims   = engineCtx->mEngine->getTensorShape(name);
        auto        dtype  = engineCtx->mEngine->getTensorDataType(name);
        bool        input  = (engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        size_t      typesz = getDataTypeSize(dtype);

        // Validate dimensions for input tensors
        if (input) {
            if (std::any_of(dims.d, dims.d + dims.nbDims, [](int val) { return val == -1; })) {
                throw std::runtime_error("Dynamic dimensions not supported.");
            }
            batch  = dims.d[0];
            height = dims.d[2];
            width  = dims.d[3];
        }

        // Calculate the tensor size in bytes
        int64_t bytes = calculateVolume(dims) * typesz;
        tensorInfos.emplace_back(name, dims, input, typesz, bytes);
    }
}

// Create CUDA graph for inference.
void DeployCGDet::createGraph() {
    // Set tensor addresses for the engine context
    for (auto& tensorInfo : tensorInfos) {
        engineCtx->mContext->setTensorAddress(tensorInfo.name.data(), tensorInfo.tensor.device(tensorInfo.bytes));
        if (!tensorInfo.input) {
            tensorInfo.tensor.host(tensorInfo.bytes);
        }
    }

    // Perform an initial inference to ensure everything is set up correctly
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 before graph creation");
    }
    CUDA(cudaStreamSynchronize(inferStream));

    // Begin capturing the CUDA graph
    CUDA(cudaStreamBeginCapture(inferStream, cudaStreamCaptureModeGlobal));

    // Copy image data to device memory if CUDA memory optimization is not enabled
    if (!cudaMem) {
        CUDA(cudaMemcpyAsync(imageTensor->device(), imageTensor->host(), inputSize * sizeof(uint8_t) * batch, cudaMemcpyHostToDevice, inferStream));
    }

    // Warp affine transformations for batched inputs
    if (batch > 1) {
        for (int i = 0; i < batch; i++) {
            CUDA(cudaEventRecord(inputEvents[i * 2], inferStream));
            CUDA(cudaStreamWaitEvent(inputStreams[i], inputEvents[i * 2]));

            uint8_t* input  = static_cast<uint8_t*>(imageTensor->device()) + i * inputSize * sizeof(uint8_t);
            float*   output = static_cast<float*>(tensorInfos[0].tensor.device()) + i * inputSize;
            cudaWarpAffine(input, width, height, output, width, height, transforms[i].matrix, inputStreams[i]);

            CUDA(cudaEventRecord(inputEvents[i * 2 + 1], inputStreams[i]));
            CUDA(cudaStreamWaitEvent(inferStream, inputEvents[i * 2 + 1]));
        }
    } else {
        cudaWarpAffine(static_cast<uint8_t*>(imageTensor->device()), width, height, static_cast<float*>(tensorInfos[0].tensor.device()), width, height, transforms[0].matrix, inferStream);
    }

    // Enqueue the inference operation
    if (!engineCtx->mContext->enqueueV3(inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 during graph creation");
    }

    // Copy the output data from device to host
    for (auto& tensorInfo : tensorInfos) {
        if (!tensorInfo.input) {
            CUDA(cudaMemcpyAsync(tensorInfo.tensor.host(), tensorInfo.tensor.device(), tensorInfo.bytes, cudaMemcpyDeviceToHost, inferStream));
        }
    }

    // End capturing the CUDA graph
    CUDA(cudaStreamEndCapture(inferStream, &inferGraph));

    // Instantiate the CUDA graph
    CUDA(cudaGraphInstantiate(&inferGraphExec, inferGraph, nullptr, nullptr, 0));
}

// Retrieve graph nodes for the CUDA graph.
void DeployCGDet::getGraphNodes() {
    size_t numNodes = cudaMem ? batch : batch + 1;
    graphNodes      = std::make_unique<cudaGraphNode_t[]>(numNodes);
    CUDA(cudaGraphGetNodes(inferGraph, graphNodes.get(), &numNodes));

    int idx = 0;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(graphNodes[i], &nodeType);
        if (nodeType == cudaGraphNodeTypeKernel) {
            CUDA(cudaGraphKernelNodeGetParams(graphNodes[i], &kernelsParams[idx]));
            idx++;
        } else if (nodeType == cudaGraphNodeTypeMemcpy) {
            CUDA(cudaGraphMemcpyNodeGetParams(graphNodes[i], &memcpyParams));
        }
    }
}

// Perform object detection on a single image.
DetectionResult DeployCGDet::predict(const Image& image) {
    auto results = predict(std::vector<Image>{image});
    if (results.empty()) {
        return DetectionResult();
    }
    return results[0];
}

// Perform object detection on a batch of images.
std::vector<DetectionResult> DeployCGDet::predict(const std::vector<Image>& images) {
    std::vector<DetectionResult> results;
    if (images.size() != batch) {
        std::cerr << "Error: Batch size mismatch. Expected " << batch << " images, but got " << images.size() << " images." << std::endl;
        return results;
    }

    // Update graph nodes for each image in the batch
    if (cudaMem) {
        for (int i = 0; i < batch; i++) {
            transforms[i].update(images[i].width, images[i].height, width, height);

            kernelsParams[i].kernelParams[0] = (void*)&images[i].rgbPtr;
            kernelsParams[i].kernelParams[1] = (void*)&images[i].width;
            kernelsParams[i].kernelParams[2] = (void*)&images[i].height;
            kernelsParams[i].kernelParams[6] = (void*)&transforms[i].matrix[0];
            kernelsParams[i].kernelParams[7] = (void*)&transforms[i].matrix[1];
            CUDA(cudaGraphExecKernelNodeSetParams(inferGraphExec, graphNodes[i], &kernelsParams[i]));
        }
    } else {
        int totalSize = 0;
        for (int i = 0; i < batch; i++) {
            transforms[i].update(images[i].width, images[i].height, width, height);
            imageSize[i]  = images[i].width * images[i].height * 3;
            totalSize    += imageSize[i];
        }

        void* host   = imageTensor->host(totalSize * sizeof(uint8_t));
        void* device = imageTensor->device(totalSize * sizeof(uint8_t));

        // Copy each image data to a contiguous region in host memory
        void* hostPtr = host;
        for (int i = 0; i < batch; i++) {
            std::memcpy(hostPtr, images[i].rgbPtr, imageSize[i] * sizeof(uint8_t));
            hostPtr = static_cast<void*>(static_cast<uint8_t*>(hostPtr) + imageSize[i]);
        }

        memcpyParams.srcPtr = make_cudaPitchedPtr(imageTensor->host(), totalSize * sizeof(uint8_t), totalSize * sizeof(uint8_t), 1);
        memcpyParams.dstPtr = make_cudaPitchedPtr(imageTensor->device(), totalSize * sizeof(uint8_t), totalSize * sizeof(uint8_t), 1);
        memcpyParams.extent = make_cudaExtent(totalSize * sizeof(uint8_t), 1, 1);
        CUDA(cudaGraphExecMemcpyNodeSetParams(inferGraphExec, graphNodes[0], &memcpyParams));

        uint8_t* devicePtr = static_cast<uint8_t*>(device);
        for (int i = 0; i < batch; i++) {
            kernelsParams[i].kernelParams[0] = (void*)&devicePtr;
            kernelsParams[i].kernelParams[1] = (void*)&images[i].width;
            kernelsParams[i].kernelParams[2] = (void*)&images[i].height;
            kernelsParams[i].kernelParams[6] = (void*)&transforms[i].matrix[0];
            kernelsParams[i].kernelParams[7] = (void*)&transforms[i].matrix[1];
            CUDA(cudaGraphExecKernelNodeSetParams(inferGraphExec, graphNodes[i + 1], &kernelsParams[i]));
            devicePtr += imageSize[i];
        }
    }

    // Launch the CUDA graph
    CUDA(cudaGraphLaunch(inferGraphExec, inferStream));

    // Synchronize the stream to ensure all operations are completed
    CUDA(cudaStreamSynchronize(inferStream));

    results.reserve(batch);
    for (int i = 0; i < batch; ++i) {
        results.emplace_back(postProcess(i));
    }

    return results;
}

}  // namespace deploy
