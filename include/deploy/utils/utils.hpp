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

#include <string>

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

}  // namespace deploy
