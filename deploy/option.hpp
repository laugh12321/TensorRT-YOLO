/**
 * @file option.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 定义图像预处理配置和推理选项配置的结构体
 * @date 2025-01-09
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cassert>
#include <optional>
#include <vector>
#include <vector_functions.hpp>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @brief 图像预处理配置结构体
 *
 */
struct ProcessConfig {
    bool   swap_rb      = false;                                                  // < 是否交换 R 和 B 通道
    float  border_value = 114.0f;                                                 // < 边界值，用于填充
    float3 alpha        = make_float3(1.0 / 255.0f, 1.0 / 255.0f, 1.0 / 255.0f);  // < 归一化系数
    float3 beta         = make_float3(0.0f, 0.0f, 0.0f);                          // < 偏移量

    /**
     * @brief 设置图像通道交换
     *
     * @param swap_rb
     */
    void enableSwapRB() {
        this->swap_rb = true;
    }

    /**
     * @brief 设置边界值
     *
     * @param border_value
     */
    void setBorderValue(float border_value) {
        this->border_value = border_value;
    }

    /**
     * @brief 设置归一化参数
     *
     * @param mean
     * @param std
     */
    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) {
        assert(mean.size() == 3 && std.size() == 3 && "ProcessConfig: requires the size of mean and std to be 3.");

        alpha.x = 1.0 / 255.0f / std[0];
        alpha.y = 1.0 / 255.0f / std[1];
        alpha.z = 1.0 / 255.0f / std[2];
        beta.x  = -mean[0] / std[0];
        beta.y  = -mean[1] / std[1];
        beta.z  = -mean[2] / std[2];
    }
};

/**
 * @brief 推理选项配置结构体
 *
 */
struct DEPLOYAPI InferOption {
    int                 device_id                 = 0;      // < GPU ID
    bool                cuda_mem                  = false;  // < 推理数据是否已经在 CUDA 显存中
    bool                enable_managed_memory     = false;  // < 是否启用统一内存
    bool                enable_performance_report = false;  // < 是否启用性能报告
    std::optional<int2> input_shape;                        // < 输入数据的高、宽，未设置时表示宽度可变（用于输入数据宽高确定的任务场景：监控视频分析，AI外挂等）
    ProcessConfig       config;                             // < 图像预处理配置

    /**
     * @brief 设置 GPU 设备 ID
     *
     * @param id
     */
    void setDeviceId(int id) {
        device_id = id;
    }

    /**
     * @brief 设置推理数据在 CUDA 显存中
     *
     */
    void enableCudaMem() {
        cuda_mem = true;
    }

    /**
     * @brief 是否启用统一内存
     *
     */
    void enableManagedMemory() {
        enable_managed_memory = true;
    }

    /**
     * @brief 启用性能报告
     *
     */
    void enablePerformanceReport() {
        enable_performance_report = true;
    }

    /**
     * @brief 设置图像通道交换
     *
     * @param swap_rb
     */
    void enableSwapRB() {
        config.enableSwapRB();
    }

    /**
     * @brief 设置边界值
     *
     * @param border_value
     */
    void setBorderValue(float border_value) {
        config.setBorderValue(border_value);
    }

    /**
     * @brief 设置归一化参数
     *
     * @param mean
     * @param std
     */
    void setNormalizeParams(const std::vector<float>& mean, const std::vector<float>& std) {
        config.setNormalizeParams(mean, std);
    }

    /**
     * @brief 设置输入数据的宽高，未设置时表示宽高可变。（用于输入数据宽高确定的任务场景：监控视频分析，AI外挂等）
     *
     * @param width 宽度
     * @param height 高度
     */
    void setInputDimensions(int width, int height) {
        input_shape = make_int2(height, width);
    }
};

}  // namespace deploy
