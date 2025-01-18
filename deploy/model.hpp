/**
 * @file model.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 模型定义
 * @date 2025-01-16
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "deploy/infer/backend.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"

namespace deploy {

// 基类模板
template <typename ResultType>
class DEPLOYAPI BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    BaseModel() = default;

    /**
     * @brief 构造一个新的 BaseModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
        : backend_(std::make_unique<TrtBackend>(trt_engine_file, infer_option)) {}

    virtual ~BaseModel() = default;

    /**
     * @brief 克隆 BaseModel 对象
     *
     * @return 克隆后的 BaseModel 对象的智能指针
     */
    virtual std::unique_ptr<BaseModel<ResultType>> clone() const = 0;

    /**
     * @brief 对单张图像进行推理
     *
     * @param image 输入图像
     * @return 推理结果
     */
    ResultType predict(const Image& image);

    /**
     * @brief 对多张图像进行推理
     *
     * @param images 输入图像向量
     * @return 推理结果向量
     */
    std::vector<ResultType> predict(const std::vector<Image>& images);

    /**
     * @brief 获取批量大小
     *
     * @return 批量大小
     */
    int batch_size() const;

protected:
    virtual ResultType postProcess(int idx) = 0;  // < 后处理方法，由派生类实现

    std::unique_ptr<TrtBackend> backend_;
};

// 分类模型
class DEPLOYAPI ClassifyModel : public BaseModel<ClassifyRes> {
public:
    using BaseModel<ClassifyRes>::BaseModel;                         // < 继承构造函数

    std::unique_ptr<BaseModel<ClassifyRes>> clone() const override;  // < 实现 clone 方法

protected:
    ClassifyRes postProcess(int idx) override;
};

// 检测模型
class DEPLOYAPI DetectModel : public BaseModel<DetectRes> {
public:
    using BaseModel<DetectRes>::BaseModel;                         // < 继承构造函数

    std::unique_ptr<BaseModel<DetectRes>> clone() const override;  // < 实现 clone 方法

protected:
    DetectRes postProcess(int idx) override;
};

// 旋转检测模型
class DEPLOYAPI OBBModel : public BaseModel<OBBRes> {
public:
    using BaseModel<OBBRes>::BaseModel;                         // < 继承构造函数

    std::unique_ptr<BaseModel<OBBRes>> clone() const override;  // < 实现 clone 方法

protected:
    OBBRes postProcess(int idx) override;
};

// 分割模型
class DEPLOYAPI SegmentModel : public BaseModel<SegmentRes> {
public:
    using BaseModel<SegmentRes>::BaseModel;                         // < 继承构造函数

    std::unique_ptr<BaseModel<SegmentRes>> clone() const override;  // < 实现 clone 方法

protected:
    SegmentRes postProcess(int idx) override;
};

// 姿态估计模型
class DEPLOYAPI PoseModel : public BaseModel<PoseRes> {
public:
    using BaseModel<PoseRes>::BaseModel;                         // < 继承构造函数

    std::unique_ptr<BaseModel<PoseRes>> clone() const override;  // < 实现 clone 方法

protected:
    PoseRes postProcess(int idx) override;
};

}  // namespace deploy
