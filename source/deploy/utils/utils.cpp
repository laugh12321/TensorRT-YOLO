#include <fstream>
#include <iostream>
#include <stdexcept>

#include "deploy/core/macro.hpp"
#include "deploy/utils/utils.hpp"

namespace deploy {

std::vector<char> loadFile(const std::string& filePath) {
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
    : mStream(stream) {
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
    float milliseconds = 0.0F;
    CHECK(cudaEventElapsedTime(&milliseconds, mStart, mStop));
    setMilliseconds(getMilliseconds() + milliseconds);
}

}  // namespace deploy
