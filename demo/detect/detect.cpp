#include <CLI/CLI.hpp>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>

#include "deploy/utils/utils.hpp"
#include "deploy/vision/detection.hpp"
#include "deploy/vision/result.hpp"

namespace fs = std::filesystem;

// Get image files in a directory
std::vector<std::string> getImagesInDirectory(const std::string& folderPath) {
    std::vector<std::string> imageFiles;
    for (const auto& entry : fs::directory_iterator(folderPath)) {
        const auto extension = entry.path().extension().string();
        if (fs::is_regular_file(entry) && (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp")) {
            imageFiles.push_back(entry.path().string());
        }
    }
    return imageFiles;
}

// Create output directory
void createOutputDirectory(const std::string& outputPath) {
    if (!fs::exists(outputPath) && !fs::create_directories(outputPath)) {
        throw std::runtime_error("Failed to create output directory: " + outputPath);
    } else if (!fs::is_directory(outputPath)) {
        throw std::runtime_error("Output path exists but is not a directory: " + outputPath);
    }
}

// Generate label and color pairs
std::vector<std::pair<std::string, cv::Scalar>> generateLabelColorPairs(const std::string& labelFile) {
    std::ifstream                                   file(labelFile);
    std::vector<std::pair<std::string, cv::Scalar>> labelColorPairs;
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + labelFile);
    }

    auto generateRandomColor = []() {
        std::random_device                 rd;
        std::mt19937                       gen(rd());
        std::uniform_int_distribution<int> dis(0, 255);
        return cv::Scalar(dis(gen), dis(gen), dis(gen));
    };

    std::string label;
    while (std::getline(file, label)) {
        labelColorPairs.emplace_back(label, generateRandomColor());
    }
    return labelColorPairs;
}

// Converts a bounding box with a given angle to its four corner points
std::vector<cv::Point> xyxyr2xyxyxyxy(const deploy::Box& box) {
    // Calculate the cosine and sine of the angle
    float cos_value = std::cos(box.theta);
    float sin_value = std::sin(box.theta);

    // Calculate the center coordinates of the box
    float center_x = (box.left + box.right) * 0.5f;
    float center_y = (box.top + box.bottom) * 0.5f;

    // Calculate the half width and half height of the box
    float half_width  = (box.right - box.left) * 0.5f;
    float half_height = (box.bottom - box.top) * 0.5f;

    // Calculate the rotated corner vectors
    float vec_x1 = half_width * cos_value;
    float vec_y1 = half_width * sin_value;
    float vec_x2 = half_height * sin_value;
    float vec_y2 = half_height * cos_value;

    // Return the four corners of the rotated rectangle
    return {
        cv::Point(center_x + vec_x1 - vec_x2, center_y + vec_y1 + vec_y2),
        cv::Point(center_x + vec_x1 + vec_x2, center_y + vec_y1 - vec_y2),
        cv::Point(center_x - vec_x1 + vec_x2, center_y - vec_y1 - vec_y2),
        cv::Point(center_x - vec_x1 - vec_x2, center_y - vec_y1 + vec_y2)};
}

// Visualize detection results
void visualize(cv::Mat& image, const deploy::DetectionResult& result, const std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs, bool is_obb) {
    for (size_t i = 0; i < result.num; ++i) {
        const auto& box       = result.boxes[i];
        int         cls       = result.classes[i];
        float       score     = result.scores[i];
        const auto& label     = labelColorPairs[cls].first;
        const auto& color     = labelColorPairs[cls].second;
        std::string labelText = label + " " + cv::format("%.2f", score);

        // Draw rectangle and label
        int      baseLine;
        cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);

        if (is_obb) {
            auto corners = xyxyr2xyxyxyxy(box);
            cv::polylines(image, {corners}, true, color, 2, cv::LINE_AA);
            cv::rectangle(image, cv::Point(corners[0].x, corners[0].y - labelSize.height), cv::Point(corners[0].x + labelSize.width, corners[0].y), color, -1);
            cv::putText(image, labelText, corners[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        } else {
            cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
            cv::rectangle(image, cv::Point(box.left, box.top - labelSize.height), cv::Point(box.left + labelSize.width, box.top), color, -1);
            cv::putText(image, labelText, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        }
    }
}

int main(int argc, char** argv) {
    CLI::App app{"YOLO Series Inference Script"};

    int         mode         = 0;
    bool        useCudaGraph = false;
    std::string enginePath, inputPath, outputPath, labelPath;
    app.add_option("-e,--engine", enginePath, "Serialized TensorRT engine")->required()->check(CLI::ExistingFile);
    app.add_option("-m,--mode", mode, "Mode for inference: 0 for Detection, 1 for OBB.")->required();
    app.add_option("-i,--input", inputPath, "Path to image or directory")->required()->check(CLI::ExistingPath);
    app.add_option("-o,--output", outputPath, "Directory to save results");
    app.add_option("-l,--labels", labelPath, "File to use for reading the class labels from")->check(CLI::ExistingFile);
    app.add_flag("--cudaGraph", useCudaGraph, "Optimize inference using CUDA Graphs, compatible with static models only.");

    CLI11_PARSE(app, argc, argv);

    if (mode != 0 && mode != 1) {
        std::cerr << "Error: "
                  << "Invalid mode: " + std::to_string(mode) + ". Please use 0 for Detection, 1 for OBB." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::vector<std::pair<std::string, cv::Scalar>> labels;
    if (!outputPath.empty()) {
        labels = generateLabelColorPairs(labelPath);
        createOutputDirectory(outputPath);
    }

    bool is_obb = (mode == 1);

    std::shared_ptr<deploy::BaseDet> model;
    if (useCudaGraph) {
        model = std::make_shared<deploy::DeployCGDet>(enginePath, is_obb);
    } else {
        model = std::make_shared<deploy::DeployDet>(enginePath, is_obb);
    }

    if (fs::is_regular_file(inputPath)) {
        cv::Mat cvimage = cv::imread(inputPath, cv::IMREAD_COLOR);
        cv::cvtColor(cvimage, cvimage, cv::COLOR_BGR2RGB);
        deploy::Image image(cvimage.data, cvimage.cols, cvimage.rows);
        auto          result = model->predict(image);
        if (!outputPath.empty()) {
            cv::cvtColor(cvimage, cvimage, cv::COLOR_RGB2BGR);
            visualize(cvimage, result, labels, is_obb);
            cv::imwrite(outputPath + "/" + fs::path(inputPath).filename().string(), cvimage);
        }
    } else {
        auto imageFiles = getImagesInDirectory(inputPath);
        if (imageFiles.empty()) {
            throw std::runtime_error("No image files found in the directory: " + inputPath);
        }

        int              count     = 0;
        const size_t     batchSize = model->batch;
        deploy::GpuTimer gpuTimer;
        deploy::CpuTimer cpuTimer;

        for (size_t i = 0; i < imageFiles.size(); i += batchSize) {
            std::vector<cv::Mat>       images;
            std::vector<deploy::Image> imgBatch;
            std::vector<std::string>   imgNameBatch;

            for (size_t j = i; j < i + batchSize && j < imageFiles.size(); ++j) {
                cv::Mat image = cv::imread(imageFiles[j], cv::IMREAD_COLOR);
                cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                images.emplace_back(image);
                imgBatch.emplace_back(image.data, image.cols, image.rows);
                imgNameBatch.emplace_back(fs::path(imageFiles[j]).filename().string());
            }

            if (images.size() != batchSize && useCudaGraph) break;

            if (i != 5) {
                cpuTimer.start();
                gpuTimer.start();
            }

            auto results = model->predict(imgBatch);

            if (i > 5) {
                gpuTimer.stop();
                cpuTimer.stop();
                count++;
            }

            if (!outputPath.empty()) {
                for (size_t j = 0; j < images.size(); ++j) {
                    cv::cvtColor(images[j], images[j], cv::COLOR_RGB2BGR);
                    visualize(images[j], results[j], labels, is_obb);
                    cv::imwrite(outputPath + "/" + imgNameBatch[j], images[j]);
                }
            }
        }

        if (count > 0) {
            std::cout << "Average infer CPU elapsed time: " << cpuTimer.milliseconds() / count << " ms" << std::endl;
            std::cout << "Average infer GPU elapsed time: " << gpuTimer.milliseconds() / count << " ms" << std::endl;
        }
    }

    std::cout << "Inference completed." << std::endl;
    return 0;
}