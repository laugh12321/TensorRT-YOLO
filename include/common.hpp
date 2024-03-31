#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept>

struct Box {
    float left, top, right, bottom;
};

struct DetectInfo {
    int num = 0;
    std::vector<Box> boxes{};
    std::vector<int> classes{};
    std::vector<float> scores{};
};

struct AffineTransform {
    float matrix[6];
    cv::Size lastSize;

    void calculate(const cv::Size& fromSize, const cv::Size& toSize) {
        if (fromSize == lastSize) return;
        lastSize = fromSize;

        double scale = std::min(static_cast<double>(toSize.width) / fromSize.width,
                                static_cast<double>(toSize.height) / fromSize.height);
        double offset = 0.5 * scale - 0.5;

        double scaleFromWidth = -0.5 * scale * fromSize.width;
        double scaleFromHeight = -0.5 * scale * fromSize.height;
        double halfToWidth = 0.5 * toSize.width;
        double halfToHeight = 0.5 * toSize.height;

        double invD = (scale != 0.0) ? 1.0 / (scale * scale) : 0.0;
        double A = scale * invD;

        matrix[0] = A;
        matrix[1] = 0.0;
        matrix[2] = -A * (scaleFromWidth + halfToWidth + offset);
        matrix[3] = 0.0;
        matrix[4] = A;
        matrix[5] = -A * (scaleFromHeight + halfToHeight + offset);
    }

    void apply(float x, float y, float* ox, float* oy) const {
        *ox = static_cast<float>(matrix[0] * x + matrix[1] * y + matrix[2]);
        *oy = static_cast<float>(matrix[3] * x + matrix[4] * y + matrix[5]);
    }
};

inline std::vector<char> loadFile(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Error opening file: " + filePath);
    }

    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    if (fileSize <= 0) {
        return {}; // If file size is zero, return an empty vector
    }

    std::vector<char> fileContent(fileSize);
    file.read(fileContent.data(), fileSize);

    if (!file) {
        throw std::runtime_error("Error reading file: " + filePath);
    }

    return fileContent;
}

inline void visualize(cv::Mat& image, const DetectInfo& detectInfo) {
    cv::Mat mask = cv::Mat::zeros(image.size(), CV_8UC3);

    for (size_t i = 0; i < detectInfo.num; i++) {
        Box box = detectInfo.boxes[i];
        int cls = detectInfo.classes[i];
        float score = detectInfo.scores[i];

        std::stringstream ss;
        ss << std::fixed << std::setprecision(2) << score;
        std::string label_text = std::to_string(cls) + " " + ss.str();

        // Draw rectangle with corners
        int pad = std::min((box.right - box.left) / 6, (box.bottom - box.top) / 6);
        std::vector<std::pair<cv::Point, cv::Point>> edges = {
            {cv::Point(box.left, box.top), cv::Point(box.left + pad, box.top)},
            {cv::Point(box.left, box.top), cv::Point(box.left, box.top + pad)},
            {cv::Point(box.right, box.top), cv::Point(box.right - pad, box.top)},
            {cv::Point(box.right, box.top), cv::Point(box.right, box.top + pad)},
            {cv::Point(box.left, box.bottom), cv::Point(box.left + pad, box.bottom)},
            {cv::Point(box.left, box.bottom), cv::Point(box.left, box.bottom - pad)},
            {cv::Point(box.right, box.bottom), cv::Point(box.right - pad, box.bottom)},
            {cv::Point(box.right, box.bottom), cv::Point(box.right, box.bottom - pad)}
        };
        for (const auto& edge : edges) {
            cv::line(image, edge.first, edge.second, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        // Draw label text
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, nullptr);
        cv::Point text_origin(box.left, box.top - label_size.height);
        cv::rectangle(image, text_origin + cv::Point(0, label_size.height), text_origin + cv::Point(label_size.width, 0), cv::Scalar(0, 255, 0), -1);
        cv::putText(image, label_text, text_origin + cv::Point(0, label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::rectangle(mask, text_origin + cv::Point(0, label_size.height), text_origin + cv::Point(label_size.width, 0), cv::Scalar(0, 255, 0), -1);
        cv::putText(mask, label_text, text_origin + cv::Point(0, label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        // Draw rectangle
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::rectangle(mask, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), -1);
    }

    // Add weighted mask to image
    cv::addWeighted(image, 0.8, mask, 0.2, 0, image);
}
