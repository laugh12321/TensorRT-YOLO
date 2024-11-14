#pragma once

#include "deploy/vision/inference.hpp"
#include "deploy/vision/result.hpp"
#include "nodes/vp_primary_infer_node.h"

namespace vp_nodes {

// TensorRT-YOLO detector node, based on vp_primary_infer_node
class vp_trtyolo_detector : public vp_primary_infer_node {
private:
    bool                             use_cudagraph = false;
    std::shared_ptr<deploy::BaseDet> detector      = nullptr;

protected:
    // Override: Run inference logic for the whole batch of frames
    virtual void run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

    // Override: Placeholder for postprocessing (not implemented in this class)
    virtual void postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) override;

public:
    // Constructor: Initializes the detector based on CUDA availability and batch size
    // Parameters:
    //   node_name: Name of the node
    //   model_path: Path to the model
    //   labels_path: Path to labels file (optional)
    //   cudagraph: Whether to use CUDA graph (default false)
    //   batch: Inference batch size (default 1)
    //   device_id: GPU device ID (default 0)
    vp_trtyolo_detector(std::string node_name, std::string model_path, std::string labels_path = "", bool cudagraph = false, int batch = 1, int device_id = 0);

    // Destructor: Cleans up resources
    ~vp_trtyolo_detector();
};

}  // namespace vp_nodes
