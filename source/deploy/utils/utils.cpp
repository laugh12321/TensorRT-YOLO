#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

#include "deploy/core/macro.hpp"
#include "deploy/utils/utils.hpp"

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

std::vector<std::pair<std::string, cv::Scalar>> GenerateLabelColorParis(const std::string& label_file) {
    std::vector<std::pair<std::string, cv::Scalar>> label_color_pairs;

    auto GenerateRandomColor = []() -> cv::Scalar {
        std::random_device                 rd;
        std::mt19937                       gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        return cv::Scalar(dis(gen), dis(gen), dis(gen));
    };

    std::ifstream file(label_file);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << label_file << std::endl;
        return label_color_pairs;
    }

    std::string label;
    while (std::getline(file, label)) {
        label_color_pairs.push_back(std::make_pair(label, GenerateRandomColor()));
    }

    file.close();
    return label_color_pairs;
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
