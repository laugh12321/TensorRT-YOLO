#pragma once
#include <vector>

#include "common/plugin.h"
#include "efficientPoseNMSParameters.h"

namespace nvinfer1
{
namespace plugin
{

class EfficientPoseNMSPlugin : public IPluginV2DynamicExt
{
public:
    explicit EfficientPoseNMSPlugin(EfficientPoseNMSParameters param);
    EfficientPoseNMSPlugin(void const* data, size_t length);
    ~EfficientPoseNMSPlugin() override = default;

    // IPluginV2 methods
    char const* getPluginType() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(char const* libNamespace) noexcept override;
    char const* getPluginNamespace() const noexcept override;

    // IPluginV2Ext methods
    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputType, int32_t nbInputs) const noexcept override;

    // IPluginV2DynamicExt methods
    IPluginV2DynamicExt* clone() const noexcept override;
    DimsExprs getOutputDimensions(
        int32_t outputIndex, DimsExprs const* inputs, int32_t nbInputs, IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(PluginTensorDesc const* inputs, int32_t nbInputs, PluginTensorDesc const* outputs,
        int32_t nbOutputs) const noexcept override;
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

protected:
    EfficientPoseNMSParameters mParam{};
    bool initialized{false};
    std::string mNamespace;

private:
    void deserialize(int8_t const* data, size_t length);
};

// Standard NMS Plugin Operation
class EfficientPoseNMSPluginCreator : public nvinfer1::pluginInternal::BaseCreator
{
public:
    EfficientPoseNMSPluginCreator();
    ~EfficientPoseNMSPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV2DynamicExt* createPlugin(char const* name, PluginFieldCollection const* fc) noexcept override;
    IPluginV2DynamicExt* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength) noexcept override;

protected:
    PluginFieldCollection mFC;
    EfficientPoseNMSParameters mParam;
    std::vector<PluginField> mPluginAttributes;
    std::string mPluginName;
};
} // namespace plugin
} // namespace nvinfer1