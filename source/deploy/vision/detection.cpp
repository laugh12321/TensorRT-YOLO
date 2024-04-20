#include "deploy/vision/detection.hpp"

#include <cuda_runtime_api.h>

#include <stdexcept>

#include "deploy/core/types.hpp"
#include "deploy/utils/utils.hpp"

namespace deploy {

DeployDet::DeployDet(const std::string& file) {
    auto data = LoadFile(file);
    if (data.empty()) throw std::runtime_error("File is empty.");
    if (engine_ctx_) engine_ctx_.reset();
    engine_ctx_ = std::make_shared<EngineContext>();
    if (!engine_ctx_->Construct(data.data(), data.size())) {
        throw std::runtime_error("Failed to construct engine context.");
    }

    Setup();
}

DeployDet::~DeployDet() {
    Release();
}

DetectionResult DeployDet::Predict(const cv::Mat& image) {
    auto images = {image};
    auto result = Predict(images);
    if (result.empty()) return {};
    return result[0];
}

std::vector<DetectionResult> DeployDet::Predict(
    const std::vector<cv::Mat>& images) {
    int num_image = images.size();
    if (num_image == 0 || num_image > batch_size) return {};

    // Preprocessing
    // Allocate space
    for (auto& tensor_info : tensors_info_) {
        tensor_info.dims.d[0] = num_image;
        if (tensor_info.is_dynamic) {
            tensor_info.UpdateVolume();
        }
        tensor_info.tensor.Device(tensor_info.vol);
        if (!tensor_info.is_input) {
            tensor_info.tensor.Host(tensor_info.vol);
        }
    }

    if (num_image > 1) {
        for (size_t i = 0; i < num_image; i++) {
            PreProcess(i, images[i],
                       input_streams_[i]);  // Convert stream to cudaStream_t
        }

        for (size_t i = 0; i < num_image; i++) {
            CUDA_CHECK_ERROR(cudaStreamSynchronize(input_streams_[i]));
        }
    } else {
        PreProcess(0, images[0], infer_stream_);
    }

    // Inference
    if (!Infer()) return {};

    // Copy data from device to host
    for (auto& tensor_info : tensors_info_) {
        if (!tensor_info.is_input) {
            CUDA_CHECK_ERROR(cudaMemcpyAsync(
                tensor_info.tensor.Host(), tensor_info.tensor.Device(),
                tensor_info.vol * tensor_info.type_size, cudaMemcpyDeviceToHost,
                infer_stream_));
        }
    }

    // Synchronize stream
    CUDA_CHECK_ERROR(cudaStreamSynchronize(infer_stream_));

    // Postprocessing
    std::vector<DetectionResult> results;
    results.reserve(num_image);  // Reserve space to avoid reallocation
    for (int i = 0; i < num_image; i++) {
        results.emplace_back(PostProcess(i));
    }

    return results;  // Return empty vector by default
}

void DeployDet::Release() {
    // Release allocated resources
    if (!input_streams_.empty()) {
        for (auto& stream_ : input_streams_) {
            CUDA_CHECK_ERROR(cudaStreamDestroy(stream_));
        }
        input_streams_.clear();
    }

    if (infer_stream_ != nullptr) {
        CUDA_CHECK_ERROR(cudaStreamDestroy(infer_stream_));
        infer_stream_ = nullptr;
    }

    // Clear other resources
    input_transforms_.clear();
    input_tensors_.clear();
    tensors_info_.clear();
}

void DeployDet::Allocate() {
    CUDA_CHECK_ERROR(cudaStreamCreate(&infer_stream_));
    for (size_t i = 0; i < batch_size; ++i) {
        cudaStream_t stream = nullptr;
        CUDA_CHECK_ERROR(cudaStreamCreate(&stream));
        input_streams_.emplace_back(stream);
    }
    input_transforms_.resize(batch_size, AffineTransform());
    input_tensors_.resize(batch_size, Tensor(sizeof(uint8_t)));
}

bool DeployDet::Infer() {
    // Iterate through each TensorInfo object in tensors_info_
    for (auto& tensor_info : tensors_info_) {
        // Set device address of tensor in engine context
        engine_ctx_->context->setTensorAddress(tensor_info.name.data(),
                                               tensor_info.tensor.Device());
        // If tensor is input and dynamic, set its shape in engine context
        if (tensor_info.is_input && tensor_info.is_dynamic) {
            engine_ctx_->context->setInputShape(tensor_info.name.data(),
                                                tensor_info.dims);
        }
    }

    // Enqueue inference tasks to CUDA stream and return results
    return engine_ctx_->context->enqueueV3(infer_stream_);
}

void DeployDet::Setup() {
    Release();

    int tensor_num = engine_ctx_->engine->getNbIOTensors();
    tensors_info_.reserve(tensor_num);
    for (size_t i = 0; i < tensor_num; i++) {
        const auto* name       = engine_ctx_->engine->getIOTensorName(i);
        auto        dims       = engine_ctx_->engine->getTensorShape(name);
        auto        data_type  = engine_ctx_->engine->getTensorDataType(name);
        bool        is_input   = (engine_ctx_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT);
        bool        is_dynamic = std::any_of(dims.d, dims.d + dims.nbDims,
                                             [](int val) { return val == -1; });

        if (is_input) {
            if (is_dynamic) {
                dims = engine_ctx_->engine->getProfileShape(
                    name, 0, nvinfer1::OptProfileSelector::kMAX);
            }
            batch_size = dims.d[0];
            height_    = dims.d[2];
            width_     = dims.d[3];
        } else if (!is_input && is_dynamic) {
            dims.d[0] = batch_size;
        }

        TensorInfo tensor_info(i, name, dims, is_dynamic, is_input, data_type);
        tensors_info_.emplace_back(tensor_info);
    }

    Allocate();
}

void DeployDet::PreProcess(const int idx, const cv::Mat& image,
                           cudaStream_t stream) {
    input_transforms_[idx].CalculateWarpMatrix(cv::Size(image.cols, image.rows),
                                               cv::Size(width_, height_));

    int64_t input_size   = static_cast<int64_t>(3) * height_ * width_;
    float*  input_device = static_cast<float*>(tensors_info_[0].tensor.Device()) + idx * input_size;

    int64_t image_size  = static_cast<int64_t>(3) * image.cols * image.rows;
    int64_t matrix_size = sizeof(input_transforms_[idx].warp_matrix);
    int64_t total_size  = RoundUp(image_size + matrix_size);
    auto    device_ptr  = input_tensors_[idx].Device(total_size);
    auto    host_ptr    = input_tensors_[idx].Host(total_size);

    float*   matrix_device = static_cast<float*>(device_ptr);
    float*   matrix_host   = static_cast<float*>(host_ptr);
    uint8_t* image_device  = static_cast<uint8_t*>(device_ptr) + matrix_size;
    uint8_t* image_host    = static_cast<uint8_t*>(host_ptr) + matrix_size;

    memcpy(image_host, image.data, image_size * sizeof(uint8_t));
    memcpy(matrix_host, input_transforms_[idx].warp_matrix, matrix_size);
    CUDA_CHECK_ERROR(cudaMemcpyAsync(image_device, image_host,
                                     image_size * sizeof(uint8_t),
                                     cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_ERROR(cudaMemcpyAsync(matrix_device, matrix_host, matrix_size,
                                     cudaMemcpyHostToDevice, stream));
    BilinearWarpAffine(image_device, image.cols * 3, image.cols, image.rows,
                       input_device, width_, height_, matrix_device, 128,
                       stream);
}

DetectionResult DeployDet::PostProcess(const int idx) {
    int    num     = static_cast<int*>(tensors_info_[1].tensor.Host())[idx];
    float* boxes   = static_cast<float*>(tensors_info_[2].tensor.Host()) + idx * tensors_info_[2].dims.d[1] * tensors_info_[2].dims.d[2];
    float* scores  = static_cast<float*>(tensors_info_[3].tensor.Host()) + idx * tensors_info_[3].dims.d[1];
    int*   classes = static_cast<int*>(tensors_info_[4].tensor.Host()) + idx * tensors_info_[4].dims.d[1];

    DetectionResult result;
    result.num = num;

    for (int i = 0; i < num; ++i) {
        float left   = boxes[i * 4];
        float top    = boxes[i * 4 + 1];
        float right  = boxes[i * 4 + 2];
        float bottom = boxes[i * 4 + 3];

        // Apply affine transformation
        input_transforms_[idx].ApplyWarp(left, top, &left, &top);
        input_transforms_[idx].ApplyWarp(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.emplace_back(scores[i]);
        result.classes.emplace_back(classes[i]);
    }

    return result;
}

}  // namespace deploy
