#pragma once

#include <cuda_runtime.h>

#include <iostream>

#ifdef ENABLE_DEPLOY_BUILDING_DLL
#if defined(_WIN32)
#define DEPLOY_DECL __declspec(dllexport)
#elif defined(__GNUC__) && ((__GNUC__ >= 4) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 3))
#define DEPLOY_DECL __attribute__((visibility("default")))
#else
#define DEPLOY_DECL
#endif
#else
#define DEPLOY_DECL
#endif

namespace deploy {

/**
 * @brief Checks for CUDA errors and prints error information if an error
 * occurs.
 *
 * @param code The CUDA error code to check.
 * @param file The file where the error occurred.
 * @param line The line number where the error occurred.
 * @return bool True if no CUDA error occurred, false otherwise.
 */
inline bool CheckCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error:\n";
        std::cerr << "    File:       " << file << "\n";
        std::cerr << "    Line:       " << line << "\n";
        std::cerr << "    Error code: " << code << "\n";
        std::cerr << "    Error text: " << cudaGetErrorString(code) << "\n";
        return false;
    }
    return true;
}

}  // namespace deploy

#define CUDA_CHECK_ERROR(code) deploy::CheckCudaError((code), __FILE__, __LINE__)
