/**
 * @file macro.hpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 宏定义
 * @date 2025-01-13
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <iostream>

#ifdef _MSC_VER
#define DEPLOYAPI __declspec(dllexport)
#else
#define DEPLOYAPI __attribute__((visibility("default")))
#endif

namespace deploy {

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

}  // namespace deploy