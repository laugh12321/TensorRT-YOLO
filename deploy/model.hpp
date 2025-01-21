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
#include "deploy/utils/utils.hpp"

namespace deploy {

/**
 * @brief 基类模板，用于定义模型的基本结构和接口。
 *
 * 该模板类封装了模型的推理逻辑、性能统计和后处理功能，适用于多种类型的模型。
 * 通过模板参数 `ResultType`，可以灵活地支持不同类型的推理结果。
 *
 * @tparam ResultType 推理结果的类型，例如 `ClassifyRes`、`DetectRes` 等。
 */
template <typename ResultType>
class DEPLOYAPI BaseModel {
public:
    // 私有的无参构造函数，仅在 clone 方法中使用
    BaseModel()  = default;
    ~BaseModel() = default;

    /**
     * @brief 构造一个新的 BaseModel 对象
     *
     * @param trt_engine_file TensorRT 引擎文件路径
     * @param infer_option 推理选项
     */
    explicit BaseModel(const std::string& trt_engine_file, const InferOption& infer_option)
        : backend_(std::make_unique<TrtBackend>(trt_engine_file, infer_option)) {
        if (backend_->option.enable_performance_report) {
            infer_gpu_trace_ = std::make_unique<GpuTimer>(backend_->stream);
            infer_cpu_trace_ = std::make_unique<CpuTimer>();
        }
    }

    /**
     * @brief 克隆 BaseModel 对象
     *
     * @return 克隆后的 BaseModel 对象的智能指针
     */
    std::unique_ptr<BaseModel<ResultType>> clone() const;

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
     * @brief 获取性能报告
     *
     * @return 包含吞吐量、CPU延迟和GPU延迟的元组
     */
    std::tuple<std::string, std::string, std::string> performanceReport();

    /**
     * @brief 获取批量大小
     *
     * @return 批量大小
     */
    int batch_size() const;

protected:
    /**
     * @brief 后处理方法，由派生类实现
     *
     * @param idx 索引
     * @return 后处理后的结果
     */
    ResultType postProcess(int idx);

    std::unique_ptr<TrtBackend> backend_;         // < TensorRT 后端

    unsigned long long        total_request_{0};  // < 总请求数
    std::unique_ptr<GpuTimer> infer_gpu_trace_;   // < GPU推理计时器
    std::unique_ptr<CpuTimer> infer_cpu_trace_;   // < CPU推理计时器
};

// 实例化模板类
template class BaseModel<ClassifyRes>;
template class BaseModel<DetectRes>;
template class BaseModel<OBBRes>;
template class BaseModel<SegmentRes>;
template class BaseModel<PoseRes>;

// 定义模型的别名
typedef BaseModel<ClassifyRes> ClassifyModel;
typedef BaseModel<DetectRes>   DetectModel;
typedef BaseModel<OBBRes>      OBBModel;
typedef BaseModel<SegmentRes>  SegmentModel;
typedef BaseModel<PoseRes>     PoseModel;

}  // namespace deploy
