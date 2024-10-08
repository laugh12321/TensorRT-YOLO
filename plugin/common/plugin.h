/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TRT_PLUGIN_H
#define TRT_PLUGIN_H
#include "NvInferPlugin.h"
#include "common/checkMacrosPlugin.h"
#include <cstring>
#include <cuda_runtime.h>
#include <functional>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>

// Enumerator for status
typedef enum
{
    STATUS_SUCCESS = 0,
    STATUS_FAILURE = 1,
    STATUS_BAD_PARAM = 2,
    STATUS_NOT_SUPPORTED = 3,
    STATUS_NOT_INITIALIZED = 4
} pluginStatus_t;

namespace nvinfer1
{

namespace pluginInternal
{

class BaseCreator : public IPluginCreator
{
public:
    void setPluginNamespace(char const* libNamespace) noexcept override
    {
        mNamespace = libNamespace;
    }

    char const* getPluginNamespace() const noexcept override
    {
        return mNamespace.c_str();
    }

protected:
    std::string mNamespace;
};

} // namespace pluginInternal

namespace plugin
{

// Write values into buffer
template <typename Type, typename BufferType>
void write(BufferType*& buffer, Type const& val)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    std::memcpy(buffer, &val, sizeof(Type));
    buffer += sizeof(Type);
}

// Read values from buffer
template <typename OutType, typename BufferType>
OutType read(BufferType const*& buffer)
{
    static_assert(sizeof(BufferType) == 1, "BufferType must be a 1 byte type.");
    OutType val{};
    std::memcpy(&val, static_cast<void const*>(buffer), sizeof(OutType));
    buffer += sizeof(OutType);
    return val;
}

// Check that all required field names are present in the PluginFieldCollection.
// If not, throw a PluginError with a message stating which fields are missing.
void validateRequiredAttributesExist(std::set<std::string> requiredFieldNames, PluginFieldCollection const* fc);

} // namespace plugin
} // namespace nvinfer1

#ifndef DEBUG

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#else

#define CSC(call, err)                                                                                                 \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t cudaStatus = call;                                                                                 \
        if (cudaStatus != cudaSuccess)                                                                                 \
        {                                                                                                              \
            printf("%s %d CUDA FAIL %s\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus));                        \
            return err;                                                                                                \
        }                                                                                                              \
    } while (0)

#endif // DEBUG

#endif // TRT_PLUGIN_H