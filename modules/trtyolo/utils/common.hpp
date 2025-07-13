/**
 * @file common.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 常用的工具函数、结构体和计时器类的定义
 * @date 2025-06-02
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <numeric>
#include <optional>
#include <vector>

namespace trtyolo {

/**
 * @brief 检查 CUDA 错误并处理，通过打印错误消息。
 *
 * 此函数用于验证 CUDA API 调用的结果。如果发生错误，它将输出错误详情，包括文件名、行号和错误描述。
 * 如果检测到 CUDA 错误，程序将终止。
 *
 * @param code CUDA API 调用返回的 CUDA 错误码。
 * @param file 发生错误的文件名。
 * @param line 发生错误的行号。
 */
inline void checkCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Failure at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief 宏，用于简化 CUDA 错误检查。
 *
 * 此宏封装了 `checkCudaError` 函数，使检查 CUDA API 调用错误更加方便。如果 CUDA 调用返回错误，
 * 宏会捕获发生错误的文件和行号，并输出错误消息。
 *
 * @param code 要检查错误的 CUDA API 调用。
 */
#define CHECK(code) checkCudaError((code), __FILE__, __LINE__)

/**
 * @brief 生成错误消息的宏。
 *
 * 此宏用于生成包含文件名、行号和函数名的错误消息，方便定位错误位置。
 *
 * @param msg 错误描述信息。
 */
#define MAKE_ERROR_MESSAGE(msg) (std::string("Error in ") + __FILE__ + ":" + std::to_string(__LINE__) + " (" + __FUNCTION__ + "): " + msg)

/**
 * @brief 图像预处理配置结构体
 *
 */
struct ProcessConfig {
    bool   swap_rb      = false;                                                  // < 是否交换 R 和 B 通道
    float  border_value = 114.0f;                                                 // < 边界值，用于填充
    float3 alpha        = make_float3(1.0 / 255.0f, 1.0 / 255.0f, 1.0 / 255.0f);  // < 归一化系数
    float3 beta         = make_float3(0.0f, 0.0f, 0.0f);                          // < 偏移量
};

/**
 * @brief 推理选项配置结构体
 *
 */
struct InferConfig {
    int                 device_id                 = 0;      // < GPU ID
    bool                cuda_mem                  = false;  // < 推理数据是否已经在 CUDA 显存中
    bool                enable_managed_memory     = false;  // < 是否启用统一内存
    bool                enable_performance_report = false;  // < 是否启用性能报告
    std::optional<int2> input_shape;                        // < 输入数据的高、宽，未设置时表示宽度可变（用于输入数据宽高确定的任务场景：监控视频分析，AI外挂等）
    ProcessConfig       config;                             // < 图像预处理配置
};

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
class CpuTimer : public TimerBase {
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
class GpuTimer : public TimerBase {
public:
    explicit GpuTimer(cudaStream_t stream);  // < 构造函数，传入 CUDA 流
    ~GpuTimer();                             // < 析构函数

    void start() override;                   // < 重写 start 方法，开始 GPU 计时
    void stop() override;                    // < 重写 stop 方法，停止 GPU 计时

private:
    cudaEvent_t  mStart, mStop;              // < 计时事件
    cudaStream_t mStream;                    // < CUDA 流
};

}  // namespace trtyolo