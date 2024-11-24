#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "deploy/core/types.hpp"
#include "deploy/utils/utils.hpp"
#include "deploy/vision/inference.hpp"

namespace deploy {

// Constructor to initialize BaseTemplate with a model file, optional CUDA memory flag, and device index.
template <typename T>
BaseTemplate<T>::BaseTemplate(const std::string& file, bool cudaMem, int device) : cudaMem(cudaMem) {
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

// Processes the inference results for a specific index.
template <>
DetResult BaseTemplate<DetResult>::postProcess(const int idx) {
    int    num     = static_cast<int*>(this->tensorInfos[1].tensor.host())[idx];
    float* boxes   = static_cast<float*>(this->tensorInfos[2].tensor.host()) + idx * this->tensorInfos[2].dims.d[1] * this->tensorInfos[2].dims.d[2];
    float* scores  = static_cast<float*>(this->tensorInfos[3].tensor.host()) + idx * this->tensorInfos[3].dims.d[1];
    int*   classes = static_cast<int*>(this->tensorInfos[4].tensor.host()) + idx * this->tensorInfos[4].dims.d[1];

    DetResult result;
    result.num = num;

    int boxSize = this->tensorInfos[2].dims.d[2];
    for (int i = 0; i < num; ++i) {
        float left   = boxes[i * boxSize];
        float top    = boxes[i * boxSize + 1];
        float right  = boxes[i * boxSize + 2];
        float bottom = boxes[i * boxSize + 3];

        // Apply affine transformation
        transforms[idx].transform(left, top, &left, &top);
        transforms[idx].transform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.emplace_back(scores[i]);
        result.classes.emplace_back(classes[i]);
    }

    return result;
}

template <>
OBBResult BaseTemplate<OBBResult>::postProcess(const int idx) {
    int    num     = static_cast<int*>(this->tensorInfos[1].tensor.host())[idx];
    float* boxes   = static_cast<float*>(this->tensorInfos[2].tensor.host()) + idx * this->tensorInfos[2].dims.d[1] * this->tensorInfos[2].dims.d[2];
    float* scores  = static_cast<float*>(this->tensorInfos[3].tensor.host()) + idx * this->tensorInfos[3].dims.d[1];
    int*   classes = static_cast<int*>(this->tensorInfos[4].tensor.host()) + idx * this->tensorInfos[4].dims.d[1];

    OBBResult result;
    result.num = num;

    int boxSize = this->tensorInfos[2].dims.d[2];
    for (int i = 0; i < num; ++i) {
        float left   = boxes[i * boxSize];
        float top    = boxes[i * boxSize + 1];
        float right  = boxes[i * boxSize + 2];
        float bottom = boxes[i * boxSize + 3];
        float theta  = boxes[i * boxSize + 4];

        // Apply affine transformation
        transforms[idx].transform(left, top, &left, &top);
        transforms[idx].transform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
        result.scores.emplace_back(scores[i]);
        result.classes.emplace_back(classes[i]);
    }

    return result;
}

template <>
SegResult BaseTemplate<SegResult>::postProcess(const int idx) {
    int maskHeight = this->tensorInfos[5].dims.d[2];
    int maskWidth  = this->tensorInfos[5].dims.d[3];

    int      num     = static_cast<int*>(this->tensorInfos[1].tensor.host())[idx];
    float*   boxes   = static_cast<float*>(this->tensorInfos[2].tensor.host()) + idx * this->tensorInfos[2].dims.d[1] * this->tensorInfos[2].dims.d[2];
    float*   scores  = static_cast<float*>(this->tensorInfos[3].tensor.host()) + idx * this->tensorInfos[3].dims.d[1];
    int*     classes = static_cast<int*>(this->tensorInfos[4].tensor.host()) + idx * this->tensorInfos[4].dims.d[1];
    uint8_t* masks   = static_cast<uint8_t*>(this->tensorInfos[5].tensor.host()) + idx * this->tensorInfos[5].dims.d[1] * maskHeight * maskWidth;

    SegResult result;
    result.num = num;

    int boxSize = this->tensorInfos[2].dims.d[2];
    for (int i = 0; i < num; ++i) {
        // Apply affine transformation
        float left   = boxes[i * boxSize];
        float top    = boxes[i * boxSize + 1];
        float right  = boxes[i * boxSize + 2];
        float bottom = boxes[i * boxSize + 3];

        transforms[idx].transform(left, top, &left, &top);
        transforms[idx].transform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.emplace_back(scores[i]);
        result.classes.emplace_back(classes[i]);

        Mask mask(maskWidth - 2 * transforms[idx].dw, maskHeight - 2 * transforms[idx].dh);

        // Crop the mask's edge area, applying offset to adjust the position
        int startIdx = i * maskHeight * maskWidth;
        int srcIndex = startIdx + transforms[idx].dh * maskWidth + transforms[idx].dw;
        for (int y = 0; y < mask.height; ++y) {
            std::memcpy(&mask.data[y * mask.width], masks + srcIndex, mask.width);
            srcIndex += maskWidth;
        }

        result.masks.emplace_back(std::move(mask));
    }

    return result;
}

template <>
PoseResult BaseTemplate<PoseResult>::postProcess(const int idx) {
    int nkpt = this->tensorInfos[5].dims.d[2];
    int ndim = this->tensorInfos[5].dims.d[3];

    int    num     = *(static_cast<int*>(this->tensorInfos[1].tensor.host()) + idx);
    float* boxes   = static_cast<float*>(this->tensorInfos[2].tensor.host()) + idx * this->tensorInfos[2].dims.d[1] * this->tensorInfos[2].dims.d[2];
    float* scores  = static_cast<float*>(this->tensorInfos[3].tensor.host()) + idx * this->tensorInfos[3].dims.d[1];
    int*   classes = static_cast<int*>(this->tensorInfos[4].tensor.host()) + idx * this->tensorInfos[4].dims.d[1];
    float* kpts    = static_cast<float*>(this->tensorInfos[5].tensor.host()) + idx * this->tensorInfos[5].dims.d[1] * nkpt * ndim;

    PoseResult result;
    result.num = num;

    int boxSize = this->tensorInfos[2].dims.d[2];
    for (int i = 0; i < num; ++i) {
        // Apply affine transformation
        float left   = boxes[i * boxSize];
        float top    = boxes[i * boxSize + 1];
        float right  = boxes[i * boxSize + 2];
        float bottom = boxes[i * boxSize + 3];

        transforms[idx].transform(left, top, &left, &top);
        transforms[idx].transform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.emplace_back(scores[i]);
        result.classes.emplace_back(classes[i]);

        std::vector<KeyPoint> keypoints;
        for (int j = 0; j < nkpt; ++j) {
            float x = kpts[i * nkpt * ndim + j * ndim];
            float y = kpts[i * nkpt * ndim + j * ndim + 1];
            transforms[idx].transform(x, y, &x, &y);
            keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
        }
        result.kpts.emplace_back(std::move(keypoints));
    }

    return result;
}

// Constructor to initialize DeployTemplate with a model file, optional CUDA memory flag, and device index.
template <typename T>
DeployTemplate<T>::DeployTemplate(const std::string& file, bool cudaMem, int device) : BaseTemplate<T>(file, cudaMem, device) {
    // Setup tensors based on the engine context
    this->setupTensors();

    // Allocate necessary resources
    this->allocate();
}

// Destructor for releasing allocated resources used by the DeployTemplate class.
template <typename T>
DeployTemplate<T>::~DeployTemplate() {
    this->release();
}

// Allocates required resources for inference execution.
template <typename T>
void DeployTemplate<T>::allocate() {
    // Create infer stream
    CUDA(cudaStreamCreate(&this->inferStream));

    // Create input streams
    this->inputStreams.resize(this->batch);
    for (auto& stream : this->inputStreams) {
        CUDA(cudaStreamCreate(&stream));
    }

    // Allocate transforms and image tensors
    this->transforms.resize(this->batch, TransformMatrix());
    if (!this->cudaMem) this->imageTensors.resize(this->batch, Tensor());
}

// Releases resources that were allocated for inference.
template <typename T>
void DeployTemplate<T>::release() {
    // Release infer stream
    if (this->inferStream != nullptr) {
        CUDA(cudaStreamDestroy(this->inferStream));
        this->inferStream = nullptr;
    }

    // Release input streams
    for (auto& stream : this->inputStreams) {
        if (stream != nullptr) {
            CUDA(cudaStreamDestroy(stream));
        }
    }
    this->inputStreams.clear();

    // Release other resources
    this->transforms.clear();
    this->tensorInfos.clear();
    this->engineCtx.reset();
    if (!this->cudaMem) this->imageTensors.clear();
}

// Configures input and output tensors for model inference.
template <typename T>
void DeployTemplate<T>::setupTensors() {
    int tensorNum = this->engineCtx->mEngine->getNbIOTensors();
    this->tensorInfos.reserve(tensorNum);
    for (size_t i = 0; i < tensorNum; i++) {
        const char* name   = this->engineCtx->mEngine->getIOTensorName(i);
        auto        dims   = this->engineCtx->mEngine->getTensorShape(name);
        auto        dtype  = this->engineCtx->mEngine->getTensorDataType(name);
        bool        input  = (this->engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        size_t      typesz = getDataTypeSize(dtype);

        if (input) {
            this->dynamic = std::any_of(dims.d, dims.d + dims.nbDims, [](int val) { return val == -1; });
            if (this->dynamic) dims = this->engineCtx->mEngine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            this->batch  = dims.d[0];
            this->height = dims.d[2];
            this->width  = dims.d[3];
        } else if (!input && this->dynamic) {
            dims.d[0] = this->batch;
        }

        int64_t bytes = calculateVolume(dims) * typesz;
        this->tensorInfos.emplace_back(name, dims, input, typesz, bytes);
    }
}

// Preprocesses a single image in the batch before inference.
template <typename T>
void DeployTemplate<T>::preProcess(const int idx, const Image& image, cudaStream_t stream) {
    this->transforms[idx].update(image.width, image.height, this->width, this->height);

    int64_t inputSize   = 3 * this->height * this->width;
    float*  inputDevice = static_cast<float*>(this->tensorInfos[0].tensor.device()) + idx * inputSize;

    void* imageDevice = nullptr;
    if (this->cudaMem) {
        imageDevice = image.rgbPtr;
    } else {
        int64_t imageSize = 3 * image.width * image.height;
        imageDevice       = this->imageTensors[idx].device(imageSize);
        void* imageHost   = this->imageTensors[idx].host(imageSize);

        std::memcpy(imageHost, image.rgbPtr, imageSize * sizeof(uint8_t));
        CUDA(cudaMemcpyAsync(imageDevice, imageHost, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    }

    cudaWarpAffine(static_cast<uint8_t*>(imageDevice), image.width, image.height, inputDevice, this->width, this->height, this->transforms[idx].matrix, stream);
}

// Performs inference on a single input image.
template <typename T>
T DeployTemplate<T>::predict(const Image& image) {
    auto results = this->predict(std::vector<Image>{image});
    if (results.empty()) {
        return T();
    }
    return results[0];
}

// Performs inference on a batch of input images.
template <typename T>
std::vector<T> DeployTemplate<T>::predict(const std::vector<Image>& images) {
    std::vector<T> results;
    int            numImages = images.size();
    if (numImages < 1 || numImages > this->batch) {
        std::cerr << "Error: Number of images (" << numImages << ") must be between 1 and " << this->batch << " inclusive." << std::endl;
        return results;
    }

    for (auto& tensorInfo : this->tensorInfos) {
        tensorInfo.dims.d[0] = numImages;
        if (this->dynamic) tensorInfo.update();
        if (!tensorInfo.input) tensorInfo.tensor.host(tensorInfo.bytes);

        this->engineCtx->mContext->setTensorAddress(tensorInfo.name.data(), tensorInfo.tensor.device(tensorInfo.bytes));
        if (tensorInfo.input && this->dynamic) {
            this->engineCtx->mContext->setInputShape(tensorInfo.name.data(), tensorInfo.dims);
        }
    }

    if (numImages > 1) {
        for (size_t i = 0; i < numImages; ++i) {
            this->preProcess(i, images[i], this->inputStreams[i]);
        }

        for (auto& stream : this->inputStreams) {
            CUDA(cudaStreamSynchronize(stream));
        }
    } else {
        this->preProcess(0, images[0], this->inferStream);
    }

    if (!this->engineCtx->mContext->enqueueV3(this->inferStream)) return {};

    for (auto& tensorInfo : this->tensorInfos) {
        if (!tensorInfo.input) {
            CUDA(cudaMemcpyAsync(tensorInfo.tensor.host(), tensorInfo.tensor.device(), tensorInfo.bytes, cudaMemcpyDeviceToHost, this->inferStream));
        }
    }

    CUDA(cudaStreamSynchronize(this->inferStream));

    results.reserve(numImages);
    for (int i = 0; i < numImages; ++i) {
        results.emplace_back(this->postProcess(i));
    }

    return results;
}

// Constructor to initialize DeployCGTemplate with a model file, optional CUDA memory flag, and device index.
template <typename T>
DeployCGTemplate<T>::DeployCGTemplate(const std::string& file, bool cudaMem, int device) : BaseTemplate<T>(file, cudaMem, device) {
    // Setup tensors based on the engine context
    this->setupTensors();

    // Allocate necessary resources
    this->allocate();

    // Create the CUDA graph
    this->createGraph();

    // Retrieve nodes from the CUDA graph
    this->getGraphNodes();

    // If CUDA memory optimization is enabled, reset the image tensor
    if (this->cudaMem) {
        this->imageTensor.reset();
    }
}

// Destructor for releasing allocated resources used by the DeployCGTemplate class.
template <typename T>
DeployCGTemplate<T>::~DeployCGTemplate() {
    this->release();
}

// Allocates required resources for inference execution.
template <typename T>
void DeployCGTemplate<T>::allocate() {
    // Create the main inference stream
    CUDA(cudaStreamCreate(&this->inferStream));

    // Create resources for batched inputs
    if (this->batch > 1) {
        this->inputStreams.reserve(this->batch);
        for (int i = 0; i < this->batch; i++) {
            cudaStream_t stream;
            CUDA(cudaStreamCreate(&stream));
            this->inputStreams.push_back(stream);
        }

        this->inputEvents.reserve(this->batch * 2);
        for (int i = 0; i < this->batch * 2; i++) {
            cudaEvent_t event;
            CUDA(cudaEventCreate(&event));
            this->inputEvents.push_back(event);
        }
    }

    // Allocate memory for tensors
    this->kernelsParams.reserve(this->batch);
    this->imageSize.reserve(this->batch);
    this->transforms.reserve(this->batch);
    this->imageTensor = std::make_shared<Tensor>();

    this->inputSize = this->width * this->height * 3;
    this->imageTensor->device(this->inputSize * sizeof(uint8_t) * this->batch);
    this->imageTensor->host(this->inputSize * sizeof(uint8_t) * this->batch);

    // Update transform matrices for each batch
    for (size_t i = 0; i < this->batch; i++) {
        this->transforms[i].update(this->width, this->height, this->width, this->height);
    }
}

// Releases resources that were allocated for inference.
template <typename T>
void DeployCGTemplate<T>::release() {
    // Release CUDA graph execution
    if (this->inferGraphExec != nullptr) {
        CUDA(cudaGraphExecDestroy(this->inferGraphExec));
        this->inferGraphExec = nullptr;
    }

    // Release CUDA graph
    if (this->inferGraph != nullptr) {
        CUDA(cudaGraphDestroy(this->inferGraph));
        this->inferGraph = nullptr;
    }

    // Release CUDA stream
    if (this->inferStream != nullptr) {
        CUDA(cudaStreamDestroy(this->inferStream));
        this->inferStream = nullptr;
    }

    // Release resources for batched inputs
    if (this->batch > 1) {
        // Release input streams
        for (auto& stream : this->inputStreams) {
            if (stream != nullptr) {
                CUDA(cudaStreamDestroy(stream));
            }
        }
        this->inputStreams.clear();

        // Release input events
        for (auto& event : this->inputEvents) {
            if (event != nullptr) {
                CUDA(cudaEventDestroy(event));
            }
        }
        this->inputEvents.clear();
    }

    // Release other resources
    this->kernelsParams.clear();
    this->imageSize.clear();
    this->tensorInfos.clear();
    this->transforms.clear();
    this->engineCtx.reset();
    this->imageTensor.reset();
}

// Configures input and output tensors for model inference.
template <typename T>
void DeployCGTemplate<T>::setupTensors() {
    int tensorNum = this->engineCtx->mEngine->getNbIOTensors();
    this->tensorInfos.reserve(tensorNum);
    for (size_t i = 0; i < tensorNum; i++) {
        const char* name   = this->engineCtx->mEngine->getIOTensorName(i);
        auto        dims   = this->engineCtx->mEngine->getTensorShape(name);
        auto        dtype  = this->engineCtx->mEngine->getTensorDataType(name);
        bool        input  = (this->engineCtx->mEngine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        size_t      typesz = getDataTypeSize(dtype);

        // Validate dimensions for input tensors
        if (input) {
            if (std::any_of(dims.d, dims.d + dims.nbDims, [](int val) { return val == -1; })) {
                throw std::runtime_error("Dynamic dimensions not supported.");
            }
            this->batch  = dims.d[0];
            this->height = dims.d[2];
            this->width  = dims.d[3];
        }

        // Calculate the tensor size in bytes
        int64_t bytes = calculateVolume(dims) * typesz;
        this->tensorInfos.emplace_back(name, dims, input, typesz, bytes);
    }
}

// Creates the CUDA graph for inference execution.
template <typename T>
void DeployCGTemplate<T>::createGraph() {
    // Set tensor addresses for the engine context
    for (auto& tensorInfo : this->tensorInfos) {
        this->engineCtx->mContext->setTensorAddress(tensorInfo.name.data(), tensorInfo.tensor.device(tensorInfo.bytes));
        if (!tensorInfo.input) {
            tensorInfo.tensor.host(tensorInfo.bytes);
        }
    }

    // Perform an initial inference to ensure everything is set up correctly
    if (!this->engineCtx->mContext->enqueueV3(this->inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 before graph creation");
    }
    CUDA(cudaStreamSynchronize(this->inferStream));

    // Begin capturing the CUDA graph
    CUDA(cudaStreamBeginCapture(this->inferStream, cudaStreamCaptureModeGlobal));

    // Copy image data to device memory if CUDA memory optimization is not enabled
    if (!this->cudaMem) {
        CUDA(cudaMemcpyAsync(this->imageTensor->device(), this->imageTensor->host(), this->inputSize * sizeof(uint8_t) * this->batch, cudaMemcpyHostToDevice, this->inferStream));
    }

    // Warp affine transformations for batched inputs
    if (this->batch > 1) {
        for (int i = 0; i < this->batch; i++) {
            CUDA(cudaEventRecord(this->inputEvents[i * 2], this->inferStream));
            CUDA(cudaStreamWaitEvent(this->inputStreams[i], this->inputEvents[i * 2], 0));

            uint8_t* input  = static_cast<uint8_t*>(this->imageTensor->device()) + i * this->inputSize * sizeof(uint8_t);
            float*   output = static_cast<float*>(this->tensorInfos[0].tensor.device()) + i * this->inputSize;
            cudaWarpAffine(input, this->width, this->height, output, this->width, this->height, this->transforms[i].matrix, this->inputStreams[i]);

            CUDA(cudaEventRecord(this->inputEvents[i * 2 + 1], this->inputStreams[i]));
            CUDA(cudaStreamWaitEvent(this->inferStream, this->inputEvents[i * 2 + 1], 0));
        }
    } else {
        cudaWarpAffine(static_cast<uint8_t*>(this->imageTensor->device()), this->width, this->height, static_cast<float*>(this->tensorInfos[0].tensor.device()), this->width, this->height, this->transforms[0].matrix, this->inferStream);
    }

    // Enqueue the inference operation
    if (!this->engineCtx->mContext->enqueueV3(this->inferStream)) {
        throw std::runtime_error("Failed to enqueueV3 during graph creation");
    }

    // Copy the output data from device to host
    for (auto& tensorInfo : this->tensorInfos) {
        if (!tensorInfo.input) {
            CUDA(cudaMemcpyAsync(tensorInfo.tensor.host(), tensorInfo.tensor.device(), tensorInfo.bytes, cudaMemcpyDeviceToHost, this->inferStream));
        }
    }

    // End capturing the CUDA graph
    CUDA(cudaStreamEndCapture(this->inferStream, &this->inferGraph));

    // Instantiate the CUDA graph
    CUDA(cudaGraphInstantiate(&this->inferGraphExec, this->inferGraph, nullptr, nullptr, 0));
}

// Retrieves and stores nodes in the CUDA graph.
template <typename T>
void DeployCGTemplate<T>::getGraphNodes() {
    size_t numNodes  = this->cudaMem ? this->batch : this->batch + 1;
    this->graphNodes = std::make_unique<cudaGraphNode_t[]>(numNodes);
    CUDA(cudaGraphGetNodes(this->inferGraph, this->graphNodes.get(), &numNodes));

    int idx = 0;
    for (size_t i = 0; i < numNodes; i++) {
        cudaGraphNodeType nodeType;
        cudaGraphNodeGetType(this->graphNodes[i], &nodeType);
        if (nodeType == cudaGraphNodeTypeKernel) {
            CUDA(cudaGraphKernelNodeGetParams(this->graphNodes[i], &this->kernelsParams[idx]));
            idx++;
        } else if (nodeType == cudaGraphNodeTypeMemcpy) {
            CUDA(cudaGraphMemcpyNodeGetParams(this->graphNodes[i], &this->memcpyParams));
        }
    }
}

// Performs inference on a single input image.
template <typename T>
T DeployCGTemplate<T>::predict(const Image& image) {
    auto results = this->predict(std::vector<Image>{image});
    if (results.empty()) {
        return T();
    }
    return results[0];
}

// Performs inference on a batch of input images.
template <typename T>
std::vector<T> DeployCGTemplate<T>::predict(const std::vector<Image>& images) {
    std::vector<T> results;
    if (images.size() != this->batch) {
        std::cerr << "Error: Batch size mismatch. Expected " << this->batch << " images, but got " << images.size() << " images." << std::endl;
        return results;
    }

    // Update graph nodes for each image in the batch
    if (this->cudaMem) {
        for (int i = 0; i < this->batch; i++) {
            this->transforms[i].update(images[i].width, images[i].height, this->width, this->height);

            this->kernelsParams[i].kernelParams[0] = (void*)&images[i].rgbPtr;
            this->kernelsParams[i].kernelParams[1] = (void*)&images[i].width;
            this->kernelsParams[i].kernelParams[2] = (void*)&images[i].height;
            this->kernelsParams[i].kernelParams[6] = (void*)&this->transforms[i].matrix[0];
            this->kernelsParams[i].kernelParams[7] = (void*)&this->transforms[i].matrix[1];
            CUDA(cudaGraphExecKernelNodeSetParams(this->inferGraphExec, this->graphNodes[i], &this->kernelsParams[i]));
        }
    } else {
        int totalSize = 0;
        for (int i = 0; i < this->batch; i++) {
            this->transforms[i].update(images[i].width, images[i].height, this->width, this->height);
            this->imageSize[i]  = images[i].width * images[i].height * 3;
            totalSize          += this->imageSize[i];
        }

        void* host   = this->imageTensor->host(totalSize * sizeof(uint8_t));
        void* device = this->imageTensor->device(totalSize * sizeof(uint8_t));

        // Copy each image data to a contiguous region in host memory
        void* hostPtr = host;
        for (int i = 0; i < this->batch; i++) {
            std::memcpy(hostPtr, images[i].rgbPtr, this->imageSize[i] * sizeof(uint8_t));
            hostPtr = static_cast<void*>(static_cast<uint8_t*>(hostPtr) + this->imageSize[i]);
        }

        this->memcpyParams.srcPtr = make_cudaPitchedPtr(this->imageTensor->host(), totalSize * sizeof(uint8_t), totalSize * sizeof(uint8_t), 1);
        this->memcpyParams.dstPtr = make_cudaPitchedPtr(this->imageTensor->device(), totalSize * sizeof(uint8_t), totalSize * sizeof(uint8_t), 1);
        this->memcpyParams.extent = make_cudaExtent(totalSize * sizeof(uint8_t), 1, 1);
        CUDA(cudaGraphExecMemcpyNodeSetParams(this->inferGraphExec, this->graphNodes[0], &this->memcpyParams));

        uint8_t* devicePtr = static_cast<uint8_t*>(device);
        for (int i = 0; i < this->batch; i++) {
            this->kernelsParams[i].kernelParams[0] = (void*)&devicePtr;
            this->kernelsParams[i].kernelParams[1] = (void*)&images[i].width;
            this->kernelsParams[i].kernelParams[2] = (void*)&images[i].height;
            this->kernelsParams[i].kernelParams[6] = (void*)&this->transforms[i].matrix[0];
            this->kernelsParams[i].kernelParams[7] = (void*)&this->transforms[i].matrix[1];
            CUDA(cudaGraphExecKernelNodeSetParams(this->inferGraphExec, this->graphNodes[i + 1], &this->kernelsParams[i]));
            devicePtr += this->imageSize[i];
        }
    }

    // Launch the CUDA graph
    CUDA(cudaGraphLaunch(this->inferGraphExec, this->inferStream));

    // Synchronize the stream to ensure all operations are completed
    CUDA(cudaStreamSynchronize(this->inferStream));

    results.reserve(this->batch);
    for (int i = 0; i < this->batch; ++i) {
        results.emplace_back(this->postProcess(i));
    }

    return results;
}

}  // namespace deploy
