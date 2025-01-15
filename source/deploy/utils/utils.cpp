/**
 * @file utils.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 提供一些实用的工具函数
 * @date 2025-01-15
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <fstream>
#include <iostream>
#include <stdexcept>

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

}  // namespace deploy
