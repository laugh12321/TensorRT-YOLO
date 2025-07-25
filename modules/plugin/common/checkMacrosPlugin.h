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
#ifndef CHECK_MACROS_PLUGIN_H
#define CHECK_MACROS_PLUGIN_H

#include "NvInfer.h"
#include <mutex>
#include <sstream>

#ifdef _MSC_VER
#define FN_NAME __FUNCTION__
#else
#define FN_NAME __func__
#endif

namespace nvinfer1
{
namespace plugin
{
template <ILogger::Severity kSeverity>
class LogStream : public std::ostream
{
    class Buf : public std::stringbuf
    {
    public:
        int32_t sync() override;
    };

    Buf buffer;
    std::mutex mLogStreamMutex;

public:
    std::mutex& getMutex()
    {
        return mLogStreamMutex;
    }
    LogStream()
        : std::ostream(&buffer){};
};

// Use mutex to protect multi-stream write to buffer
template <ILogger::Severity kSeverity, typename T>
LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, T const& msg)
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << msg;
    return stream;
}

// Special handling static numbers
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, int32_t num)
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << num;
    return stream;
}

// Special handling std::endl
template <ILogger::Severity kSeverity>
inline LogStream<kSeverity>& operator<<(LogStream<kSeverity>& stream, std::ostream& (*f)(std::ostream&) )
{
    std::lock_guard<std::mutex> guard(stream.getMutex());
    auto& os = static_cast<std::ostream&>(stream);
    os << f;
    return stream;
}

extern LogStream<ILogger::Severity::kERROR> gLogError;
extern LogStream<ILogger::Severity::kWARNING> gLogWarning;
extern LogStream<ILogger::Severity::kINFO> gLogInfo;
extern LogStream<ILogger::Severity::kVERBOSE> gLogVerbose;

void reportValidationFailure(char const* msg, char const* file, int32_t line);
void reportAssertion(char const* msg, char const* file, int32_t line);
void logError(char const* msg, char const* file, char const* fn, int32_t line);

[[noreturn]] void throwCudaError(
    char const* file, char const* function, int32_t line, int32_t status, char const* msg = nullptr);
[[noreturn]] void throwPluginError(
    char const* file, char const* function, int32_t line, int32_t status, char const* msg = nullptr);

class TRTException : public std::exception
{
public:
    TRTException(char const* fl, char const* fn, int32_t ln, int32_t st, char const* msg, char const* nm)
        : file(fl)
        , function(fn)
        , line(ln)
        , status(st)
        , message(msg)
        , name(nm)
    {
    }
    virtual void log(std::ostream& logStream) const;
    void setMessage(char const* msg)
    {
        message = msg;
    }

protected:
    char const* file{nullptr};
    char const* function{nullptr};
    int32_t line{0};
    int32_t status{0};
    char const* message{nullptr};
    char const* name{nullptr};
};

class CudaError : public TRTException
{
public:
    CudaError(char const* fl, char const* fn, int32_t ln, int32_t stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Cuda")
    {
    }
};

class PluginError : public TRTException
{
public:
    PluginError(char const* fl, char const* fn, int32_t ln, int32_t stat, char const* msg = nullptr)
        : TRTException(fl, fn, ln, stat, msg, "Plugin")
    {
    }
};

inline void caughtError(std::exception const& e)
{
    gLogError << e.what() << std::endl;
}
} // namespace plugin

} // namespace nvinfer1

#define PLUGIN_CHECK_CUDA(call)                                                                                        \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t status = call;                                                                                     \
        if (status != cudaSuccess)                                                                                     \
        {                                                                                                              \
            return status;                                                                                             \
        }                                                                                                              \
    } while (0)

#define PLUGIN_CUASSERT(status_)                                                                                       \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != cudaSuccess)                                                                                         \
        {                                                                                                              \
            const char* msg = cudaGetErrorString(s_);                                                                  \
            nvinfer1::plugin::throwCudaError(__FILE__, FN_NAME, __LINE__, s_, msg);                                    \
        }                                                                                                              \
    }

#define GET_MACRO(_1, _2, NAME, ...) NAME
#define PLUGIN_VALIDATE(...) GET_MACRO(__VA_ARGS__, PLUGIN_VALIDATE_MSG, PLUGIN_VALIDATE_DEFAULT, )(__VA_ARGS__)

// Logs failed condition and throws a PluginError.
// PLUGIN_ASSERT will eventually perform this function, at which point PLUGIN_VALIDATE
// will be removed.
#define PLUGIN_VALIDATE_DEFAULT(condition)                                                                             \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, #condition);                            \
        }                                                                                                              \
    }

#define PLUGIN_VALIDATE_MSG(condition, msg)                                                                            \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg);                                   \
        }                                                                                                              \
    }

// Logs failed assertion and aborts.
// Aborting is undesirable and will be phased-out from the plugin module, at which point
// PLUGIN_ASSERT will perform the same function as PLUGIN_VALIDATE.
#define PLUGIN_ASSERT(assertion)                                                                                       \
    {                                                                                                                  \
        if (!(assertion))                                                                                              \
        {                                                                                                              \
            nvinfer1::plugin::reportAssertion(#assertion, __FILE__, __LINE__);                                         \
        }                                                                                                              \
    }

#define PLUGIN_FAIL(msg)                                                                                               \
    {                                                                                                                  \
        nvinfer1::plugin::reportAssertion(msg, __FILE__, __LINE__);                                                    \
    }

#define PLUGIN_ERROR(msg)                                                                                              \
    {                                                                                                                  \
        nvinfer1::plugin::throwPluginError(__FILE__, FN_NAME, __LINE__, 0, msg);                                       \
    }

#define PLUGIN_CUERROR(status_)                                                                                        \
    {                                                                                                                  \
        auto s_ = status_;                                                                                             \
        if (s_ != 0)                                                                                                   \
            nvinfer1::plugin::logError(#status_ " failure.", __FILE__, FN_NAME, __LINE__);                             \
    }

#endif /*CHECK_MACROS_PLUGIN_H*/