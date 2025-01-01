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
 * @brief Checks for CUDA errors and handles them by printing an error message.
 *
 * This function verifies the result of CUDA API calls. If an error occurs, it outputs
 * the error details, including the file name, line number, and a description of the error.
 * The program will terminate if a CUDA error is detected.
 *
 * @param code The CUDA error code returned by a CUDA API call.
 * @param file The name of the file where the error occurred.
 * @param line The line number where the error occurred.
 */
inline void checkCudaError(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Failure at " << file << ":" << line << ": "
                  << cudaGetErrorString(code) << std::endl;
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief Macro to simplify CUDA error checking.
 *
 * This macro wraps around the `checkCudaError` function, making it easier to check
 * CUDA API calls for errors. If the CUDA call returns an error, the macro captures the
 * file and line where the error occurred and outputs the error message.
 *
 * @param code The CUDA API call to check for errors.
 */
#define CHECK(code) checkCudaError((code), __FILE__, __LINE__)

}  // namespace deploy