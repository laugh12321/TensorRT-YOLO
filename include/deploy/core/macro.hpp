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
 * @brief Checks for CUDA errors and prints error information if an error occurs.
 *
 * This function checks the CUDA error code returned by CUDA API calls. If the code
 * indicates an error, it prints out information about the error, including the file
 * and line number where the error occurred, the error code itself, and a textual
 * description of the error.
 *
 * @param code The CUDA error code to check.
 * @param file The file where the error occurred.
 * @param line The line number where the error occurred.
 * @return bool Returns true if no CUDA error occurred, false otherwise.
 */
inline bool cudaError(cudaError_t code, const char* file, int line) {
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

/**
 * @brief Macro for simplified CUDA error checking.
 *
 * This macro wraps the `cudaError` function, allowing easy and concise checking
 * of CUDA API calls. It evaluates the given CUDA API call `code`, and if it returns
 * an error, the `cudaError` function is called to print error information.
 *
 * @param code The CUDA API call to be executed and checked for errors.
 */
#define CUDA(code) cudaError((code), __FILE__, __LINE__)

}  // namespace deploy