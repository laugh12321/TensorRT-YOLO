#include "deploy/vision/visualize.hpp"

namespace deploy {

void Visualize(cv::Mat& image, const DetectionResult& result,
               const std::vector<std::pair<std::string, cv::Scalar>>& label_color_pairs) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);
    for (size_t i = 0; i < result.num; i++) {
        Box         box   = result.boxes[i];
        int         cls   = result.classes[i];
        float       score = result.scores[i];
        std::string label = label_color_pairs[cls].first;
        cv::Scalar  color = label_color_pairs[cls].second;

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        std::string label_text = label + " " + ss.str();

        // Draw rectangle with corners
        int pad =
            std::min((box.right - box.left) / 6, (box.bottom - box.top) / 6);
        std::vector<cv::Point> corners = {
            cv::Point(box.left, box.top), cv::Point(box.right, box.top),
            cv::Point(box.right, box.bottom), cv::Point(box.left, box.bottom)};
        for (size_t j = 0; j < corners.size(); j++) {
            cv::line(image, corners[j], corners[(j + 1) % corners.size()],
                     color, 2, cv::LINE_AA);
        }

        // Draw label text background
        cv::Size label_size = cv::getTextSize(
            label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
        cv::Point text_origin(box.left, box.top - label_size.height);
        cv::rectangle(image, text_origin + cv::Point(0, label_size.height),
                      text_origin + cv::Point(label_size.width, 0),
                      color, -1);
        cv::rectangle(mask, text_origin + cv::Point(0, label_size.height),
                      text_origin + cv::Point(label_size.width, 0),
                      color, -1);

        // Draw label text
        cv::putText(image, label_text,
                    text_origin + cv::Point(0, label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);
        cv::putText(mask, label_text,
                    text_origin + cv::Point(0, label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1,
                    cv::LINE_AA);

        // Draw rectangle
        cv::rectangle(image, cv::Point(box.left, box.top),
                      cv::Point(box.right, box.bottom), color,
                      1, cv::LINE_AA);
        cv::rectangle(mask, cv::Point(box.left, box.top),
                      cv::Point(box.right, box.bottom), color,
                      -1);
    }

    // Add weighted mask to image
    cv::addWeighted(image, 0.8, mask, 0.2, 0, image);
}

}  // namespace deploy