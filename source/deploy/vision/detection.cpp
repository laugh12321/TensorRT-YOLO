#include <algorithm>
#include <stdexcept>

#include "deploy/core/types.hpp"
#include "deploy/utils/utils.hpp"
#include "deploy/vision/detection.hpp"


namespace deploy {

DeployDet::DeployDet(const std::string& file, bool cudaMem, int device) : cudaMem(cudaMem) {
    CUDA(cudaSetDevice(device));

    release();
    auto data = loadFile(file);

    // Reset engine context
    engineCtx = std::make_shared<EngineContext>();
    if (!engineCtx->construct(data.data(), data.size())) {
        throw std::runtime_error("Failed to construct engine context.");
    }

    setupTensors();
    allocate();
}

DeployDet::~DeployDet() {
    release();
}

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

DetectionResult DeployDet::predict(const Image& image) {
    auto images = {image};
    auto result = predict(images);
    if (result.empty()) return {};
    return result[0];
}

std::vector<DetectionResult> DeployDet::predict(const std::vector<Image>& images) {
    int numImages = images.size();
    if (numImages == 0 || numImages > batch) return {};

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

    std::vector<DetectionResult> results;
    results.reserve(numImages);
    for (int i = 0; i < numImages; ++i) {
        results.emplace_back(postProcess(i));
    }

    return results;
}

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

DetectionResult DeployDet::postProcess(const int idx) {
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

}  // namespace deploy
