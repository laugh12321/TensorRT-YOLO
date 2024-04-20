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
 * @param labels A vector of strings representing class labels.
 */
void DEPLOY_DECL Visualize(cv::Mat& image, const DetectionResult& result,
                           const std::vector<std::string>& labels);

}  // namespace deploy
