#pragma once

#include "deploy/model.hpp"
#include "nodes/vp_primary_infer_node.h"

namespace vp_nodes {

// TensorRT-YOLO detector node, based on vp_primary_infer_node
class vp_trtyolo_detector : public vp_primary_infer_node {
private:
    std::shared_ptr<deploy::DetectModel> detector = nullptr;

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
    //   batch: Inference batch size (default 1)
    //   device_id: GPU device ID (default 0)
    vp_trtyolo_detector(std::string node_name, std::string model_path, std::string labels_path = "", int batch = 1, int device_id = 0);

    // Destructor: Cleans up resources
    ~vp_trtyolo_detector();
};

}  // namespace vp_nodes
