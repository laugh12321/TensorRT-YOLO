#pragma once
#include "common/plugin.h"

namespace nvinfer1 {
namespace plugin {

struct EfficientPoseNMSParameters {
    // Related to NMS Options
    float   iouThreshold           = 0.5F;
    float   scoreThreshold         = 0.5F;
    int32_t numOutputBoxes         = 100;
    int32_t numOutputBoxesPerClass = -1;
    bool    padOutputBoxesPerClass = false;
    int32_t backgroundClass        = -1;
    bool    scoreSigmoid           = false;
    int32_t boxCoding              = 0;
    bool    classAgnostic          = false;

    // Related to NMS Internals
    int32_t numSelectedBoxes  = 4096;
    int32_t scoreBits         = -1;

    // Related to Tensor Configuration
    // (These are set by the various plugin configuration methods, no need to define them during plugin creation.)
    int32_t            batchSize        = -1;
    int32_t            numClasses       = 1;
    int32_t            numBoxElements   = -1;
    int32_t            numScoreElements = -1;
    int32_t            numAnchors       = -1;
    bool               shareLocation    = true;
    bool               shareAnchors     = true;
    bool               boxDecoder       = false;
    nvinfer1::DataType datatype         = nvinfer1::DataType::kFLOAT;
};

}  // namespace plugin
}  // namespace nvinfer1
