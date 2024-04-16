#include <memory>
#include "detection.hpp"
#include "cuda_utils.hpp"


// Compute coordinates in the source image
__device__ void computeSrcCoords(float dx, float dy, float *matrix, int srcWidth, int srcHeight, float &srcX, float &srcY) {
    srcX = matrix[0] * dx + matrix[1] * dy + matrix[2];
    srcY = matrix[3] * dx + matrix[4] * dy + matrix[5];
}

// Get pixel value using bilinear interpolation
__device__ float bilinearInterpolation(uint8_t *src, int srcLineSize, int srcWidth, int srcHeight, float srcX, float srcY, int channel) {
    int x_low = floorf(srcX);
    int y_low = floorf(srcY);
    int x_high = x_low + 1;
    int y_high = y_low + 1;

    // Clamp coordinates to image boundaries
    x_low = max(0, min(x_low, srcWidth - 1));
    x_high = max(0, min(x_high, srcWidth - 1));
    y_low = max(0, min(y_low, srcHeight - 1));
    y_high = max(0, min(y_high, srcHeight - 1));

    // Compute interpolation weights
    float lx = srcX - x_low;
    float ly = srcY - y_low;
    float hx = 1.0f - lx;
    float hy = 1.0f - ly;

    // Compute pixel pointers
    uint8_t *v1 = src + y_low * srcLineSize + x_low * 3 + channel;
    uint8_t *v2 = src + y_low * srcLineSize + x_high * 3 + channel;
    uint8_t *v3 = src + y_high * srcLineSize + x_low * 3 + channel;
    uint8_t *v4 = src + y_high * srcLineSize + x_high * 3 + channel;

    // Perform bilinear interpolation
    return hy * (hx * (*v1) + lx * (*v2)) + ly * (hx * (*v3) + lx * (*v4));
}

__global__ void warpAffineBilinearKernel(
    uint8_t *src, int srcLineSize, int srcWidth, int srcHeight, float *dst, int dstWidth,
    int dstHeight, float *matrix, uint8_t constValue) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    int area = dstWidth * dstHeight;
    int tid = threadIdx.x + threadIdx.y * blockDim.x;

    // Load matrix into shared memory
    __shared__ float sharedMatrix[6];
    if (tid < 6) {
        sharedMatrix[tid] = matrix[tid];
    }
    __syncthreads();

    if (dx >= dstWidth || dy >= dstHeight) return;

    float srcX, srcY;
    computeSrcCoords(dx, dy, sharedMatrix, srcWidth, srcHeight, srcX, srcY);

    // If out of range, set to constant value
    float3 color = make_float3(constValue, constValue, constValue);

    // Precompute interpolation coefficients
    if (srcX > -1 && srcX < srcWidth && srcY > -1 && srcY < srcHeight) {
        // Perform bilinear interpolation
        color.x = bilinearInterpolation(src, srcLineSize, srcWidth, srcHeight, srcX, srcY, 0);
        color.y = bilinearInterpolation(src, srcLineSize, srcWidth, srcHeight, srcX, srcY, 1);
        color.z = bilinearInterpolation(src, srcLineSize, srcWidth, srcHeight, srcX, srcY, 2);
    }

    // Swap B and R channels
    float temp = color.z;
    color.z = color.x;
    color.x = temp;

    // Store color values directly
    float *pDstC0 = dst + dy * dstWidth + dx;
    float *pDstC1 = pDstC0 + area;
    float *pDstC2 = pDstC1 + area;

    *pDstC0 = color.x / 255.0f;
    *pDstC1 = color.y / 255.0f;
    *pDstC2 = color.z / 255.0f;
}

namespace yolo {

Detection::~Detection() {
    this->release();
    this->_engineCtx.reset();
}

bool Detection::load(const std::string& filePath) {
    auto data = loadFile(filePath);
    // Check if file content is empty or file size is zero
    if (data.empty()) {
        std::cerr << "File content is empty or file size is zero." << std::endl;
        return false;
    }

    // If there's an old engine context object, release it
    if (this->_engineCtx) {
        this->_engineCtx.reset(); // Release the old engine context object
    }

    // Create a new engine context object
    this->_engineCtx = std::make_shared<cuda_utils::EngineContext>();
    // Check if the engine context object is successfully constructed
    if (!this->_engineCtx->construct(data.data(), data.size())) {
        std::cerr << "Failed to construct engine context." << std::endl;
        return false;
    }

    // Setup the engine
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

    // Preprocessing
    // Allocate space
    for (auto& tensorInfo : this->_tensorInfos) {
        tensorInfo.dims.d[0] = numImage;
        if (tensorInfo.isDynamic) tensorInfo.updateVolume(); // Update volume for dynamic tensors
        tensorInfo.memory->device(tensorInfo.vol); // Ensure memory allocation on device
        if (!tensorInfo.isInput) tensorInfo.memory->host(tensorInfo.vol);
    }

    if (numImage > 1) {
        for (size_t i = 0; i < numImage; i++) {
            this->preprocess(i, images[i], this->_streams[i]); // Convert stream to cudaStream_t
        }

        for (size_t i = 0; i < numImage; i++) {
            cuda_utils::CUDA_CHECK_ERROR(cudaStreamSynchronize(this->_streams[i]));
        }
    } else {
        this->preprocess(0, images[0], stream);
    }

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

    // Postprocessing
    std::vector<DetectInfo> detectInfos;
    detectInfos.reserve(numImage); // Reserve space to avoid reallocation
    for (int i = 0; i < numImage; i++) {
        detectInfos.emplace_back(this->postprocess(i));
    }

    return detectInfos; // Return empty vector by default
}

void Detection::release() {
    // Release CUDA stream objects
    for (auto& stream : this->_streams) {
        cuda_utils::CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
    }
    this->_streams.clear();

    // Release other resources
    this->_tensorInfos.clear();
    this->_preBuffers.clear();
    this->_affineTransforms.clear();
}

void Detection::allocate() {
    // Allocate resources
    this->_affineTransforms.resize(batchSize, AffineTransform());
    this->_preBuffers.resize(batchSize, cuda_utils::HostDeviceMem(sizeof(uint8_t)));
    for (int i = 0; i < batchSize; ++i) {
        cudaStream_t stream;
        cuda_utils::CUDA_CHECK_ERROR(cudaStreamCreate(&stream));
        this->_streams.push_back(stream);
    }
}

void Detection::setup() {
    // Release any existing resources
    this->release();

    // Get tensor names and initialize
    int nbTensors = this->_engineCtx->engine->getNbIOTensors();
    this->_tensorInfos.resize(nbTensors);
    for (int i = 0; i < nbTensors; ++i) {
        // Directly modify tensorInfo elements
        this->_tensorInfos[i] = createTensorInfo(i);
    }

    // Allocate resources
    this->allocate();
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

    // Update volume of tensor
    tensorInfo.updateVolume();

    return tensorInfo;
}

bool Detection::infer(cudaStream_t stream) {
    // Iterate through each TensorInfo object in _tensorInfos
    for (auto& tensorInfo : this->_tensorInfos) {
        // Set device address of tensor in engine context
        this->_engineCtx->context->setTensorAddress(tensorInfo.name, tensorInfo.memory->device());
        // If tensor is input and dynamic, set its shape in engine context
        if (tensorInfo.isInput && tensorInfo.isDynamic) {
            this->_engineCtx->context->setInputShape(tensorInfo.name, tensorInfo.dims);
        }
    }

    // Enqueue inference tasks to CUDA stream and return results
    return this->_engineCtx->context->enqueueV3(stream);
}

void Detection::preprocess(const int idx, const cv::Mat& image, cudaStream_t stream) {
    // Calculate affine transformation
    this->_affineTransforms[idx].calculate(cv::Size(image.cols, image.rows), cv::Size(this->_inputWidth, this->_inputHeight));

    // Calculate input size
    int64_t inputSize = static_cast<int64_t>(3) * this->_inputHeight * this->_inputWidth;

    // Allocate device space for input tensor
    float *inputDevice = static_cast<float*>(this->_tensorInfos[0].memory->device()) + idx * inputSize;

    // Calculate sizes of image and matrix
    int64_t imageSize = static_cast<int64_t>(3) * image.rows * image.cols;
    int64_t matrixSize = cuda_utils::upbound(sizeof(this->_affineTransforms[idx].matrix));

    // Allocate memory for image and matrix on host and device
    auto deviceSpace = this->_preBuffers[idx].device(imageSize + matrixSize);
    auto hostSpace = this->_preBuffers[idx].host(imageSize + matrixSize);

    // Pointers to image and matrix on host and device
    float *matrixDevice = static_cast<float*>(deviceSpace);
    uint8_t *imageDevice = static_cast<uint8_t*>(deviceSpace) + matrixSize;
    float *matrixHost = static_cast<float*>(hostSpace);
    uint8_t *imageHost = static_cast<uint8_t*>(hostSpace) + matrixSize;

    // Copy image data and matrix from host to device
    memcpy(imageHost, image.data, imageSize * sizeof(uint8_t));
    memcpy(matrixHost, this->_affineTransforms[idx].matrix, sizeof(this->_affineTransforms[idx].matrix));
    cuda_utils::CUDA_CHECK_ERROR(cudaMemcpyAsync(imageDevice, imageHost, imageSize * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
    cuda_utils::CUDA_CHECK_ERROR(cudaMemcpyAsync(matrixDevice, matrixHost, sizeof(this->_affineTransforms[idx].matrix), cudaMemcpyHostToDevice, stream));

    dim3 blockSize(16, 16);
    dim3 gridSize((this->_inputWidth + blockSize.x - 1) / blockSize.x, (this->_inputHeight + blockSize.y - 1) / blockSize.y);
    // Perform affine transformation
    warpAffineBilinearKernel<<<gridSize, blockSize, 0, stream>>>(
        imageDevice, image.cols * 3, image.cols, image.rows, 
        inputDevice, this->_inputWidth, this->_inputHeight,
        matrixDevice, 128
    );
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

        // Apply affine transformation
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
