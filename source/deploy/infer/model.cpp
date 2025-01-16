/**
 * @file model.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 模型实现
 * @date 2025-01-16
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <cstdint>
#include <cstring>

#include "deploy/infer/model.hpp"
#include "deploy/infer/result.hpp"

namespace deploy {

template <typename ResultType>
std::vector<ResultType> BaseModel<ResultType>::predict(const std::vector<Image>& images) {
    backend_->infer(images);  // 调用推理方法

    // 预分配结果空间
    std::vector<ResultType> results(images.size());
    for (auto idx = 0u; idx < images.size(); ++idx) {
        results[idx] = postProcess(idx);
    }

    return results;
}

template <typename ResultType>
ResultType BaseModel<ResultType>::predict(const Image& image) {
    return predict(std::vector<Image>{image}).front();
}

template <typename ResultType>
int BaseModel<ResultType>::batch_size() const {
    return backend_->max_shape.x;
}

// ClassifyModel 的后处理方法实现
ClassifyRes ClassifyModel::postProcess(int idx) {
    auto&  tensor_info = backend_->tensor_infos[1];
    float* topk        = static_cast<float*>(tensor_info.buffer->host()) + idx * tensor_info.shape.d[1] * tensor_info.shape.d[2];

    ClassifyRes result;
    result.num = tensor_info.shape.d[1];
    result.scores.reserve(result.num);
    result.classes.reserve(result.num);

    for (int i = 0; i < result.num; ++i) {
        result.scores.push_back(topk[i * tensor_info.shape.d[2]]);
        result.classes.push_back(topk[i * tensor_info.shape.d[2] + 1]);
    }

    return result;
}

// DetectModel 的后处理方法实现
DetectRes DetectModel::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];

    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

    DetectRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);
    }

    return result;
}

// OBBModel 的后处理方法实现
OBBRes OBBModel::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];

    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];

    OBBRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];
        float theta = boxes[base_index + 4];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(RotatedBox{left, top, right, bottom, theta});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);
    }

    return result;
}

// SegmentModel 的后处理方法实现
SegmentRes SegmentModel::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];
    auto& mask_tensor  = backend_->tensor_infos[5];
    int   mask_height  = mask_tensor.shape.d[2];
    int   mask_width   = mask_tensor.shape.d[3];

    int      num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float*   boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float*   scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*     classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    uint8_t* masks   = static_cast<uint8_t*>(mask_tensor.buffer->host()) + idx * mask_tensor.shape.d[1] * mask_height * mask_width;

    SegmentRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.masks.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        Mask mask(mask_width - 2 * affine_transform.dst_offset_x, mask_height - 2 * affine_transform.dst_offset_y);

        // Crop the mask's edge area, applying offset to adjust the position
        int start_idx = i * mask_height * mask_width;
        int src_idx   = start_idx + affine_transform.dst_offset_y * mask_width + affine_transform.dst_offset_x;
        for (int y = 0; y < mask.height; ++y) {
            std::memcpy(&mask.data[y * mask.width], masks + src_idx, mask.width);
            src_idx += mask_width;
        }

        result.masks.emplace_back(std::move(mask));
    }

    return result;
}

// PoseModel 的后处理方法实现
PoseRes PoseModel::postProcess(int idx) {
    auto& num_tensor   = backend_->tensor_infos[1];
    auto& box_tensor   = backend_->tensor_infos[2];
    auto& score_tensor = backend_->tensor_infos[3];
    auto& class_tensor = backend_->tensor_infos[4];
    auto& kpt_tensor   = backend_->tensor_infos[5];
    int   nkpt         = kpt_tensor.shape.d[2];
    int   ndim         = kpt_tensor.shape.d[3];

    int    num     = static_cast<int*>(num_tensor.buffer->host())[idx];
    float* boxes   = static_cast<float*>(box_tensor.buffer->host()) + idx * box_tensor.shape.d[1] * box_tensor.shape.d[2];
    float* scores  = static_cast<float*>(score_tensor.buffer->host()) + idx * score_tensor.shape.d[1];
    int*   classes = static_cast<int*>(class_tensor.buffer->host()) + idx * class_tensor.shape.d[1];
    float* kpts    = static_cast<float*>(kpt_tensor.buffer->host()) + idx * kpt_tensor.shape.d[1] * nkpt * ndim;

    PoseRes result;
    result.num   = num;
    int box_size = box_tensor.shape.d[2];

    auto& affine_transform = backend_->option.input_shape.has_value()
                                 ? backend_->affine_transforms.front()
                                 : backend_->affine_transforms[idx];

    result.boxes.reserve(num);
    result.scores.reserve(num);
    result.classes.reserve(num);
    result.kpts.reserve(num);

    for (int i = 0; i < num; ++i) {
        int   base_index = i * box_size;
        float left = boxes[base_index], top = boxes[base_index + 1];
        float right = boxes[base_index + 2], bottom = boxes[base_index + 3];

        affine_transform.applyTransform(left, top, &left, &top);
        affine_transform.applyTransform(right, bottom, &right, &bottom);

        result.boxes.emplace_back(Box{left, top, right, bottom});
        result.scores.push_back(scores[i]);
        result.classes.push_back(classes[i]);

        std::vector<KeyPoint> keypoints;
        for (int j = 0; j < nkpt; ++j) {
            float x = kpts[i * nkpt * ndim + j * ndim];
            float y = kpts[i * nkpt * ndim + j * ndim + 1];
            affine_transform.applyTransform(x, y, &x, &y);
            keypoints.emplace_back((ndim == 2) ? KeyPoint(x, y) : KeyPoint(x, y, kpts[i * nkpt * ndim + j * ndim + 2]));
        }
        result.kpts.emplace_back(std::move(keypoints));
    }

    return result;
}

// ClassifyModel 的 clone 方法实现
std::unique_ptr<BaseModel<ClassifyRes>> ClassifyModel::clone() const {
    auto clone_model      = std::make_unique<ClassifyModel>();
    clone_model->backend_ = backend_->clone();  // < 克隆 TrtBackend
    return clone_model;
}

// DetectModel 的 clone 方法实现
std::unique_ptr<BaseModel<DetectRes>> DetectModel::clone() const {
    auto clone_model      = std::make_unique<DetectModel>();
    clone_model->backend_ = backend_->clone();  // < 克隆 TrtBackend
    return clone_model;
}

// OBBModel 的 clone 方法实现
std::unique_ptr<BaseModel<OBBRes>> OBBModel::clone() const {
    auto clone_model      = std::make_unique<OBBModel>();
    clone_model->backend_ = backend_->clone();  // < 克隆 TrtBackend
    return clone_model;
}

// SegmentModel 的 clone 方法实现
std::unique_ptr<BaseModel<SegmentRes>> SegmentModel::clone() const {
    auto clone_model      = std::make_unique<SegmentModel>();
    clone_model->backend_ = backend_->clone();  // < 克隆 TrtBackend
    return clone_model;
}

// PoseModel 的 clone 方法实现
std::unique_ptr<BaseModel<PoseRes>> PoseModel::clone() const {
    auto clone_model      = std::make_unique<PoseModel>();
    clone_model->backend_ = backend_->clone();  // < 克隆 TrtBackend
    return clone_model;
}

// 实例化模板类
template class BaseModel<ClassifyRes>;
template class BaseModel<DetectRes>;
template class BaseModel<OBBRes>;
template class BaseModel<SegmentRes>;
template class BaseModel<PoseRes>;

}  // namespace deploy
