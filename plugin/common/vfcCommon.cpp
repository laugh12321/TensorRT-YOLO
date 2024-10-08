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

#include "common/vfcCommon.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "efficientRotatedNMSPlugin/efficientRotatedNMSPlugin.h"
#include <vector>
#include <mutex>

using namespace nvinfer1;
using nvinfer1::plugin::EfficientRotatedNMSPluginCreator;

namespace nvinfer1
{
namespace plugin
{

class ThreadSafeLoggerFinder
{
private:
    ILoggerFinder* mLoggerFinder{nullptr};
    std::mutex mMutex;

public:
    ThreadSafeLoggerFinder() = default;

    //! Set the logger finder.
    void setLoggerFinder(ILoggerFinder* finder)
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder == nullptr && finder != nullptr)
        {
            mLoggerFinder = finder;
        }
    }

    //! Get the logger.
    ILogger* getLogger() noexcept
    {
        std::lock_guard<std::mutex> lk(mMutex);
        if (mLoggerFinder != nullptr)
        {
            return mLoggerFinder->findLogger();
        }
        return nullptr;
    }
};

ThreadSafeLoggerFinder gLoggerFinder;

ILogger* getPluginLogger()
{
    return gLoggerFinder.getLogger();
}

} // namespace plugin
} // namespace 

extern "C" TENSORRTAPI void setLoggerFinder(nvinfer1::ILoggerFinder* finder)
{
    nvinfer1::plugin::gLoggerFinder.setLoggerFinder(finder);
}

#if (TENSORRT_VERSION >= 100000)
extern "C" TENSORRTAPI IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static EfficientRotatedNMSPluginCreator efficientRotatedNMSPluginCreator;
    static IPluginCreatorInterface* const kPLUGIN_CREATOR_LIST[] = {&efficientRotatedNMSPluginCreator};
    return kPLUGIN_CREATOR_LIST;

}
#else
extern "C" TENSORRTAPI IPluginCreator* const* getPluginCreators(int32_t& nbCreators)
{
    nbCreators = 1;
    static EfficientRotatedNMSPluginCreator efficientRotatedNMSPluginCreator;
    static IPluginCreator* const kPLUGIN_CREATOR_LIST[] = {&efficientRotatedNMSPluginCreator};
    return kPLUGIN_CREATOR_LIST;
}
#endif