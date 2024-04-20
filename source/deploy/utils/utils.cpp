#include "deploy/utils/utils.hpp"

#include <fstream>
#include <iostream>
#include <stdexcept>

#include "deploy/core/macro.hpp"

namespace deploy {

std::vector<char> LoadFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);  // Open file in binary mode
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    // Get file size
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    // If file is empty, return an empty vector
    if (fileSize <= 0) {
        return {};
    }

    // Read file content into vector
    std::vector<char> fileContent(fileSize);
    file.read(fileContent.data(), fileSize);

    // Check for read errors
    if (!file) {
        throw std::runtime_error("Error reading file: " + filePath);
    }

    return fileContent;
}

GpuTimer::GpuTimer(cudaStream_t stream)
    : m_stream(stream) {
    CUDA_CHECK_ERROR(cudaEventCreate(&m_start));
    CUDA_CHECK_ERROR(cudaEventCreate(&m_stop));
}

GpuTimer::~GpuTimer() {
    CUDA_CHECK_ERROR(cudaEventDestroy(m_start));
    CUDA_CHECK_ERROR(cudaEventDestroy(m_stop));
}

void GpuTimer::Start() {
    CUDA_CHECK_ERROR(cudaEventRecord(m_start, m_stream));
}

void GpuTimer::Stop() {
    CUDA_CHECK_ERROR(cudaEventRecord(m_stop, m_stream));
    CUDA_CHECK_ERROR(cudaEventSynchronize(m_stop));
    float milliseconds = 0.0F;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&milliseconds, m_start, m_stop));
    SetMilliseconds(GetMilliseconds() + milliseconds);
}

}  // namespace deploy
