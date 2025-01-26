/**
 * @file utils.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 提供一些实用的工具函数
 * @date 2025-01-15
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */
#include <algorithm>
#include <fstream>

#include "deploy/core/macro.hpp"
#include "deploy/utils/utils.hpp"

namespace deploy {

void ReadBinaryFromFile(const std::string& file, std::string* contents) {
    std::ifstream fin(file, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        throw std::runtime_error("Failed to open file: " + file + " to read.");
    }
    fin.seekg(0, std::ios::end);
    contents->clear();
    contents->resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(contents->at(0)), contents->size());
    fin.close();
}

bool SupportsIntegratedZeroCopy(const int gpu_id) {
    // 查询设备属性，检查是否为集成显卡
    cudaDeviceProp cuprops;
    CHECK(cudaGetDeviceProperties(&cuprops, gpu_id));

    // 只有在集成显卡且支持映射主机内存时，才支持零拷贝
    if (cuprops.integrated && cuprops.canMapHostMemory) {
        return true;
    } else {
        return false;
    }
}

float findPercentile(float percentile, std::vector<float> const& timings) {
    int32_t const all     = static_cast<int32_t>(timings.size());
    int32_t const exclude = static_cast<int32_t>((1 - percentile / 100) * all);
    if (timings.empty()) {
        return std::numeric_limits<float>::infinity();
    }
    if (percentile < 0.F || percentile > 100.F) {
        throw std::runtime_error("percentile is not in [0, 100]!");
    }
    return timings[std::max(all - 1 - exclude, 0)];
}

float findMedian(std::vector<float> const& timings) {
    if (timings.empty()) {
        return std::numeric_limits<float>::infinity();
    }

    int32_t const m = timings.size() / 2;
    if (timings.size() % 2) {
        return timings[m];
    }

    return (timings[m - 1] + timings[m]) / 2;
}

PerformanceResult getPerformanceResult(std::vector<float> const& timings, std::vector<float> const& percentiles) {
    auto metricComparator  = [](float const& a, float const& b) { return a < b; };
    auto metricAccumulator = [](float acc, float const& a) { return acc + a; };

    std::vector<float> newTimings = timings;
    std::sort(newTimings.begin(), newTimings.end(), metricComparator);
    PerformanceResult result;
    result.min    = newTimings.front();
    result.max    = newTimings.back();
    result.mean   = std::accumulate(newTimings.begin(), newTimings.end(), 0.0F, metricAccumulator) / newTimings.size();
    result.median = findMedian(newTimings);
    for (auto percentile : percentiles) {
        result.percentiles.emplace_back(findPercentile(percentile, newTimings));
    }
    return result;
}

void CpuTimer::start() {
    mStart = std::chrono::high_resolution_clock::now();
}

void CpuTimer::stop() {
    mStop = std::chrono::high_resolution_clock::now();
    mMs.push_back(std::chrono::duration<float, std::milli>{mStop - mStart}.count());
}

GpuTimer::GpuTimer(cudaStream_t stream) : mStream(stream) {
    CHECK(cudaEventCreate(&mStart));
    CHECK(cudaEventCreate(&mStop));
}

GpuTimer::~GpuTimer() {
    CHECK(cudaEventDestroy(mStart));
    CHECK(cudaEventDestroy(mStop));
}

void GpuTimer::start() {
    CHECK(cudaEventRecord(mStart, mStream));
}

void GpuTimer::stop() {
    CHECK(cudaEventRecord(mStop, mStream));
    CHECK(cudaEventSynchronize(mStop));
    float ms{0.0F};
    CHECK(cudaEventElapsedTime(&ms, mStart, mStop));
    mMs.push_back(ms);
}

}  // namespace deploy
