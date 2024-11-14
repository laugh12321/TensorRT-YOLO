#include "vp_trtyolo_detector.h"

namespace vp_nodes {

vp_trtyolo_detector::vp_trtyolo_detector(std::string node_name, std::string model_path, std::string labels_path, bool cudagraph, int batch, int device_id)
    : vp_primary_infer_node(node_name, "", "", labels_path, 0, 0, batch), use_cudagraph(cudagraph) {
    // Initialize detector based on CUDA graph usage
    if (use_cudagraph) {
        detector = std::make_shared<deploy::DeployCGDet>(model_path, device_id);
        if (detector->batch != batch) {
            throw std::runtime_error("Batch size mismatch: expected " + std::to_string(batch) + ", but got " + std::to_string(detector->batch));
        }
    } else {
        detector = std::make_shared<deploy::DeployDet>(model_path, device_id);
        if (detector->batch < batch) {
            throw std::runtime_error("Batch size too large: expected <= " + std::to_string(detector->batch) + ", but got " + std::to_string(batch));
        }
    }

    this->initialized();  // Mark node as initialized
}

vp_trtyolo_detector::~vp_trtyolo_detector() {
    // Destructor: Clean up any resources
    deinitialized();  // Mark node as deinitialized
}

void vp_trtyolo_detector::run_infer_combinations(const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
    if (use_cudagraph)
        assert(frame_meta_with_batch.size() == detector->batch);  // Assert batch size consistency if using CUDA graph

    std::vector<cv::Mat>       mats_to_infer;
    std::vector<deploy::Image> images_to_infer;

    auto start_time = std::chrono::system_clock::now();  // Start time for performance measurement

    // Prepare data for inference (same as base class)
    vp_primary_infer_node::prepare(frame_meta_with_batch, mats_to_infer);
    std::transform(mats_to_infer.begin(), mats_to_infer.end(), std::back_inserter(images_to_infer), [](cv::Mat& mat) {
        return deploy::Image(mat.data, mat.cols, mat.rows);
    });

    auto prepare_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

    start_time = std::chrono::system_clock::now();

    // Perform inference on prepared images
    std::vector<deploy::DetResult> detection_results = detector->predict(images_to_infer);

    // Process detection results and update frame metadata
    for (int i = 0; i < detection_results.size(); i++) {
        auto& frame_meta       = frame_meta_with_batch[i];
        auto& detection_result = detection_results[i];

        for (int j = 0; j < detection_result.num; j++) {
            int  x      = static_cast<int>(detection_result.boxes[j].left);
            int  y      = static_cast<int>(detection_result.boxes[j].top);
            int  width  = static_cast<int>(detection_result.boxes[j].right - detection_result.boxes[j].left);
            int  height = static_cast<int>(detection_result.boxes[j].bottom - detection_result.boxes[j].top);
            auto label  = labels.size() == 0 ? "" : labels[detection_result.classes[j]];

            // Create target and update back into frame meta
            auto target = std::make_shared<vp_objects::vp_frame_target>(
                x, y, width, height, detection_result.classes[j], detection_result.scores[j],
                frame_meta->frame_index, frame_meta->channel_index, label);

            frame_meta->targets.push_back(target);
        }
    }

    auto infer_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time);

    // Cannot calculate preprocess time and postprocess time, set 0 by default.
    vp_infer_node::infer_combinations_time_cost(mats_to_infer.size(), prepare_time.count(), 0, infer_time.count(), 0);
}

void vp_trtyolo_detector::postprocess(const std::vector<cv::Mat>& raw_outputs, const std::vector<std::shared_ptr<vp_objects::vp_frame_meta>>& frame_meta_with_batch) {
    // Placeholder for postprocessing logic if needed in future enhancements
    // Currently not implemented in this class
}

}  // namespace vp_nodes
