#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "deploy/core/macro.hpp"
#include "deploy/vision/result.hpp"

namespace deploy {

/**
 * @brief Visualizes the detection results on an image.
 *
 * This function draws bounding boxes and labels on the input image based on the detection results.
 *
 * @param image The input image.
 * @param result The detection result containing information about detected objects.
 * @param label_color_pairs A vector of pairs, where each pair consists of a label (string) and a color (cv::Scalar).
 */
DEPLOY_DECL void Visualize(cv::Mat& image, const DetectionResult& result,
                           const std::vector<std::pair<std::string, cv::Scalar>>& label_color_pairs);

}  // namespace deploy
