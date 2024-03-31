#include "detection.hpp"
#include "cuda_utils.hpp"


__global__ void warpAffineBilinearKernel(
    uint8_t *src, int srcLineSize, int srcWidth, int srcHeight, float *dst, int dstWidth,
    int dstHeight, float *matrix, uint8_t constValue) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    if (dx >= dstWidth || dy >= dstHeight) return;

    float srcX = matrix[0] * dx + matrix[1] * dy + matrix[2];
    float srcY = matrix[3] * dx + matrix[4] * dy + matrix[5];

    // If out of range, set to constant value
    float c0 = constValue, c1 = constValue, c2 = constValue;

    // Precompute interpolation coefficients
    float xFractionalPart = 0.0f, yFractionalPart = 0.0f;
    int xLow = 0, yLow = 0, srcIdx = 0, srcIdx2 = 0;
    uint8_t *v1 = nullptr, *v2 = nullptr, *v3 = nullptr, *v4 = nullptr;
    if (srcX > -1 && srcX < srcWidth && srcY > -1 && srcY < srcHeight) {
        xLow = floorf(srcX);
        yLow = floorf(srcY);
        xFractionalPart = srcX - xLow;
        yFractionalPart = srcY - yLow;

        // Calculate indices and pixel values for bilinear interpolation
        srcIdx = yLow * srcLineSize + xLow * 3;
        srcIdx2 = srcIdx + srcLineSize;
        v1 = &src[srcIdx];
        v2 = &src[srcIdx + 3];
        v3 = &src[srcIdx2];
        v4 = &src[srcIdx2 + 3];

        // Bilinear interpolation
        c0 = (1.0f - yFractionalPart) * ((1.0f - xFractionalPart) * v1[0] + xFractionalPart * v2[0]) + 
             yFractionalPart * ((1.0f - xFractionalPart) * v3[0] + xFractionalPart * v4[0]);
        c1 = (1.0f - yFractionalPart) * ((1.0f - xFractionalPart) * v1[1] + xFractionalPart * v2[1]) + 
             yFractionalPart * ((1.0f - xFractionalPart) * v3[1] + xFractionalPart * v4[1]);
        c2 = (1.0f - yFractionalPart) * ((1.0f - xFractionalPart) * v1[2] + xFractionalPart * v2[2]) + 
             yFractionalPart * ((1.0f - xFractionalPart) * v3[2] + xFractionalPart * v4[2]);
    }

    // Swap B and R channels
    float temp = c2;
    c2 = c0;
    c0 = temp;

    // Normalize values to range [0, 1]
    c0 /= 255.0f;
    c1 /= 255.0f;
    c2 /= 255.0f;

    // Reorder RGB to RRRGGGBBB
    int area = dstWidth * dstHeight;
    float *pDstC0 = dst + dy * dstWidth + dx;
    float *pDstC1 = pDstC0 + area;
    float *pDstC2 = pDstC1 + area;

    *pDstC0 = c0;
    *pDstC1 = c1;
    *pDstC2 = c2;
}


namespace yolo {

bool Detection::load(const std::string& filePath) {
    auto data = loadFile(filePath);
    // Check if file content is empty or file size is zero
    if (data.empty()) {
        std::cerr << "File content is empty or file size is zero." << std::endl;
        return false;
    }

    // Release the old engine context object if it exists
    if (this->_engineCtx) {
        this->_engineCtx.reset(); // Release the old engine context object
    }

    // Create a new engine context object
    this->_engineCtx = std::make_shared<cuda_utils::EngineContext>();
    // Check if the engine context object was successfully constructed
    if (!this->_engineCtx->construct(data.data(), data.size())) {
        std::cerr << "Failed to construct engine context." << std::endl;
        return false;
    }

    // Set up the engine
    this->setup();
    return true;
}

DetectInfo Detection::predict(const cv::Mat& image, cudaStream_t stream) {
    std::vector<cv::Mat> images = { image };
    auto result = predict(images, stream);
    if (result.empty()) return {};
    return result[0];
}

std::vector<DetectInfo> Detection::predict(const std::vector<cv::Mat>& images, cudaStream_t stream) {
    int numImage = images.size();
    if (numImage == 0 || numImage > this->batchSize) return {};

    // Preprocess
    // Allocate space
    for (auto& tensorInfo : this->_tensorInfos) {
        tensorInfo.dims.d[0] = numImage;
        if (tensorInfo.isDynamic) tensorInfo.updateVolume(); // Update volume if dynamic
        tensorInfo.memory->device(tensorInfo.vol); // Ensure memory is allocated on device
        if (!tensorInfo.isInput) tensorInfo.memory->host(tensorInfo.vol);
    }

    // Ensure preBuffers size
    if (this->_preBuffers.size() < numImage) {
        this->_preBuffers.resize(numImage, std::make_shared<cuda_utils::HostDeviceMem>(sizeof(uint8_t)));
    }
    if (this->_affineTransforms.size() < numImage) {
        this->_affineTransforms.resize(numImage, AffineTransform());
    }

    // Preprocess each image
    for (size_t i = 0; i < numImage; i++)
        this->preprocess(i, images[i], stream); // Cast stream to cudaStream_t

    // Inference
    if (!this->infer(stream)) return {};

    // Copy data from device to host
    for (auto& tensorInfo : this->_tensorInfos) {
        if (!tensorInfo.isInput) {
            cuda_utils::CUDA_CHECK_ERROR(cudaMemcpyAsync(tensorInfo.memory->host(), tensorInfo.memory->device(), tensorInfo.vol * cuda_utils::dataTypeSize(tensorInfo.dataType), cudaMemcpyDeviceToHost, stream));
        }
    }

    // Synchronize stream
    cuda_utils::CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    // Postprocess
    std::vector<DetectInfo> detectInfos;
    detectInfos.reserve(numImage); // Reserve space to avoid reallocations
    for (int i = 0; i < numImage; i++) {
        detectInfos.emplace_back(this->postprocess(i));
    }

    return detectInfos; // Return an empty vector by default
}

void Detection::setup() {
    // Clear existing tensor information
    this->_tensorInfos.clear();

    // Get tensor name
    int nbTensors = this->_engineCtx->engine->getNbIOTensors();
    this->_tensorInfos.reserve(nbTensors);
    for (int i = 0; i < nbTensors; ++i) {
        // Create tensor info directly and move it into the vector
        this->_tensorInfos.emplace_back(createTensorInfo(i));
    }
}

TensorInfo Detection::createTensorInfo(int index) {
    TensorInfo tensorInfo;
    tensorInfo.index = index;
    tensorInfo.name = this->_engineCtx->engine->getIOTensorName(index);
    tensorInfo.dims = this->_engineCtx->engine->getTensorShape(tensorInfo.name);
    tensorInfo.dataType = this->_engineCtx->engine->getTensorDataType(tensorInfo.name);
    tensorInfo.memory = std::make_shared<cuda_utils::HostDeviceMem>(cuda_utils::dataTypeSize(tensorInfo.dataType));
    tensorInfo.isInput = (this->_engineCtx->engine->getTensorIOMode(tensorInfo.name) == nvinfer1::TensorIOMode::kINPUT);
    tensorInfo.isDynamic = std::any_of(tensorInfo.dims.d, tensorInfo.dims.d + tensorInfo.dims.nbDims,
        [](int val) {
            return val == -1;
        }
    );

    if (tensorInfo.isInput) {
        if (tensorInfo.isDynamic) tensorInfo.dims = this->_engineCtx->engine->getProfileShape(tensorInfo.name, 0, nvinfer1::OptProfileSelector::kMAX);
        // Update batch size, input height, and input width
        this->batchSize = tensorInfo.dims.d[0];
        this->_inputHeight = tensorInfo.dims.d[2];
        this->_inputWidth = tensorInfo.dims.d[3];
    } else if (tensorInfo.isDynamic && !tensorInfo.isInput) {
        // Set batch size for non-input tensors
        tensorInfo.dims.d[0] = this->batchSize;
    }

    // Update the volume of the tensor
    tensorInfo.updateVolume();

    return tensorInfo;
}

bool Detection::infer(cudaStream_t stream) {
    // Iterate over each TensorInfo object in _tensorInfos
    for (auto& tensorInfo : this->_tensorInfos) {
        // Set the device address for the tensor in the engine context
        this->_engineCtx->context->setTensorAddress(tensorInfo.name, tensorInfo.memory->device());
        // If the tensor is an input tensor and dynamic, set its shape in the engine context
        if (tensorInfo.isInput && tensorInfo.isDynamic) {
            this->_engineCtx->context->setInputShape(tensorInfo.name, tensorInfo.dims);
        }
    }

    // Enqueue the inference task to the CUDA stream and return the result
    return this->_engineCtx->context->enqueueV3(stream);
}

void Detection::preprocess(const int idx, const cv::Mat& image, cudaStream_t stream) {
    // Calculate affine transformation
    this->_affineTransforms[idx].calculate(cv::Size(image.cols, image.rows), cv::Size(this->_inputWidth, this->_inputHeight));

    // Calculate input size
    int64_t inputSize = static_cast<int64_t>(3) * this->_inputHeight * this->_inputWidth;

    // Allocate device space for input tensor
    float *inputDevice = static_cast<float*>(this->_tensorInfos[0].memory->device()) + idx * inputSize;

    // Calculate image and matrix size
    int64_t imageSize = static_cast<int64_t>(3) * image.rows * image.cols;
    int64_t matrixSize = cuda_utils::upbound(sizeof(this->_affineTransforms[idx].matrix));

    // Allocate memory for image and matrix on host and device
    auto deviceSpace = this->_preBuffers[idx]->device(imageSize + matrixSize);
    auto hostSpace = this->_preBuffers[idx]->host(imageSize + matrixSize);

    // Pointers for image and matrix on host and device
    float *matrixDevice = static_cast<float*>(deviceSpace);
    uint8_t *imageDevice = static_cast<uint8_t*>(deviceSpace) + matrixSize;
    float *matrixHost = static_cast<float*>(hostSpace);
    uint8_t *imageHost = static_cast<uint8_t*>(hostSpace) + matrixSize;

    // Copy image data and matrix from host to device
    memcpy(imageHost, image.data, imageSize * sizeof(uint8_t));
    memcpy(matrixHost, this->_affineTransforms[idx].matrix, sizeof(this->_affineTransforms[idx].matrix));
    cuda_utils::CUDA_CHECK_ERROR(cudaMemcpyAsync(imageDevice, imageHost, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    cuda_utils::CUDA_CHECK_ERROR(cudaMemcpyAsync(matrixDevice, matrixHost, sizeof(this->_affineTransforms[idx].matrix), cudaMemcpyHostToDevice, stream));

    // Synchronize stream
    cuda_utils::CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    dim3 blockSize(32, 32);
    dim3 gridSize((this->_inputWidth + blockSize.x - 1) / blockSize.x, (this->_inputHeight + blockSize.y - 1) / blockSize.y);
    // Perform affine transformation
    warpAffineBilinearKernel<<<gridSize, blockSize, 0, stream>>>(
        imageDevice, image.cols * 3, image.cols, image.rows, 
        inputDevice, this->_inputWidth, this->_inputHeight,
        matrixDevice, 128
    );
    cuda_utils::CUDA_CHECK_ERROR(cudaGetLastError());
}

DetectInfo Detection::postprocess(const int idx) {
    int num = reinterpret_cast<int*>(this->_tensorInfos[1].memory->host())[idx];
    float *boxes = reinterpret_cast<float*>(this->_tensorInfos[2].memory->host()) + idx * this->_tensorInfos[2].dims.d[1] * this->_tensorInfos[2].dims.d[2];
    float *scores = reinterpret_cast<float*>(this->_tensorInfos[3].memory->host()) + idx * this->_tensorInfos[3].dims.d[1];
    int *classes = reinterpret_cast<int*>(this->_tensorInfos[4].memory->host()) + idx * this->_tensorInfos[4].dims.d[1];

    DetectInfo detectInfo;
    detectInfo.num = num;

    for (int i = 0; i < num; ++i) {
        float left = boxes[i * 4];
        float top = boxes[i * 4 + 1];
        float right = boxes[i * 4 + 2];
        float bottom = boxes[i * 4 + 3];

        // Perform affine transformation
        this->_affineTransforms[idx].apply(left, top, &left, &top);
        this->_affineTransforms[idx].apply(right, bottom, &right, &bottom);

        detectInfo.boxes.emplace_back(Box{left, top, right, bottom});
        detectInfo.scores.emplace_back(scores[i]);
        detectInfo.classes.emplace_back(classes[i]);
    }

    return detectInfo;
}

std::shared_ptr<Detection> load(const std::string& filePath) {
    std::shared_ptr<Detection> detect = std::make_shared<Detection>();
    if (!detect->load(filePath)) return nullptr;
    return detect;
}

} // namespace yolo
