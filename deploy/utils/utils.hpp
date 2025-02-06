/**
 * @file utils.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 提供一些实用的工具函数
 * @date 2025-01-15
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cuda_runtime_api.h>

#include <chrono>
#include <numeric>
#include <string>
#include <vector>

#include "deploy/core/macro.hpp"

namespace deploy {

/**
 * @brief 从文件中读取二进制数据并存储到指定的字符串中
 *
 * 该函数打开指定路径的文件，读取其中的二进制数据，并将数据存储到 `contents` 字符串中。
 * 如果打开文件失败，将抛出异常。
 *
 * @param file 文件路径
 * @param contents 用于存储读取到的二进制数据的字符串
 */
void ReadBinaryFromFile(const std::string& file, std::string* contents);

/**
 * @brief 检查指定的 GPU 是否支持集成零拷贝内存
 *
 * 该函数查询 GPU 的属性，判断该 GPU 是否为集成显卡并且是否支持将主机内存映射到设备内存（零拷贝内存）。
 * 如果支持零拷贝，返回 `true`，否则返回 `false`。
 *
 * @param gpu_id GPU 的设备 ID
 * @return true 如果 GPU 支持集成零拷贝内存
 * @return false 如果 GPU 不支持集成零拷贝内存
 */
bool SupportsIntegratedZeroCopy(const int gpu_id);

/**
 * @brief 在一个升序的时间序列中找到指定的百分位数
 * @note 百分位数必须在 [0, 100] 范围内。否则，将抛出异常。
 *
 * @param percentile 百分位数
 * @param timings 时间序列
 * @return float 计算得到的百分位数值
 */
float findPercentile(float percentile, std::vector<float> const& timings);

/**
 * @brief 在一个已排序的时间序列中找到中位数
 *
 * @param timings 时间序列
 * @return float 中位数值
 */
float findMedian(std::vector<float> const& timings);

/**
 * @struct PerformanceResult
 * @brief 性能指标的性能结果
 */
struct PerformanceResult {
    float              min{0.f};     // < 最小值
    float              max{0.f};     // < 最大值
    float              mean{0.f};    // < 平均值
    float              median{0.f};  // < 中位数
    std::vector<float> percentiles;  // < 百分位数值列表
};

/**
 * @brief 获取性能结果对象
 *
 * @param timings 时间序列
 * @param percentiles 百分位数列表
 * @return PerformanceResult 性能结果对象
 */
PerformanceResult getPerformanceResult(std::vector<float> const& timings, std::vector<float> const& percentiles);

/**
 * @brief 定义一个计时器基类
 *
 * 该类提供了基本的计时功能，包括开始计时、停止计时、获取计时结果（毫秒）、重置计时结果以及获取总计时（毫秒）。
 * 它是一个抽象类，用于派生出具体的 CPU 计时器和 GPU 计时器类。
 */
class TimerBase {
public:
    virtual void       start() {}  // < 虚函数，用于开始计时
    virtual void       stop() {}   // < 虚函数，用于停止计时
    std::vector<float> milliseconds() const noexcept {
        return mMs;
    }  // < 获取计时结果（毫秒）
    void reset() noexcept {
        mMs.clear();
    }  // < 重置计时结果
    float totalMilliseconds() const noexcept {
        return std::accumulate(mMs.begin(), mMs.end(), 0.0F);
    }  // < 获取总计时（毫秒）

protected:
    std::vector<float> mMs;  // < 计时结果列表
};

/**
 * @brief 定义一个 CPU 计时器类
 *
 * 该类继承自 TimerBase 类，实现了具体的 CPU 计时功能。
 * 它使用 C++ 标准库中的高精度时钟来记录时间。
 */
class DEPLOYAPI CpuTimer : public TimerBase {
public:
    void start() override;                                                      // < 重写 start 方法，开始 CPU 计时
    void stop() override;                                                       // < 重写 stop 方法，停止 CPU 计时

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart, mStop;  // < 计时起止时间点
};  // class CpuTimer

/**
 * @brief 定义一个 GPU 计时器类
 *
 * 该类继承自 TimerBase 类，实现了具体的 GPU 计时功能。
 * 它使用 CUDA 事件来记录 GPU 上的时间。
 */
class DEPLOYAPI GpuTimer : public TimerBase {
public:
    explicit GpuTimer(cudaStream_t stream);  // < 构造函数，传入 CUDA 流
    ~GpuTimer();                             // < 析构函数

    void start() override;                   // < 重写 start 方法，开始 GPU 计时
    void stop() override;                    // < 重写 stop 方法，停止 GPU 计时

private:
    cudaEvent_t  mStart, mStop;              // < 计时事件
    cudaStream_t mStream;                    // < CUDA 流
};

}  // namespace deploy
