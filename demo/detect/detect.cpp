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

// Visualize detection results
void visualize(cv::Mat& image, const deploy::DetectionResult& result, const std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs) {
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
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - labelSize.height), cv::Point(box.left + labelSize.width, box.top), color, -1);
        cv::putText(image, labelText, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
}

int main(int argc, char** argv) {
    CLI::App app{"YOLO Series Inference Script"};

    bool        useCudaGraph = false;
    std::string enginePath, inputPath, outputPath, labelPath;
    app.add_option("-e,--engine", enginePath, "Serialized TensorRT engine")->required()->check(CLI::ExistingFile);
    app.add_option("-i,--input", inputPath, "Path to image or directory")->required()->check(CLI::ExistingPath);
    app.add_option("-o,--output", outputPath, "Directory to save results");
    app.add_option("-l,--labels", labelPath, "File to use for reading the class labels from")->check(CLI::ExistingFile);
    app.add_flag("--cudaGraph", useCudaGraph, "Optimize inference using CUDA Graphs, compatible with static models only.");

    CLI11_PARSE(app, argc, argv);

    std::vector<std::pair<std::string, cv::Scalar>> labels;
    if (!outputPath.empty()) {
        labels = generateLabelColorPairs(labelPath);
        createOutputDirectory(outputPath);
    }

    std::shared_ptr<deploy::BaseDet> model;
    if (useCudaGraph) {
        model = std::make_shared<deploy::DeployCGDet>(enginePath);
    } else {
        model = std::make_shared<deploy::DeployDet>(enginePath);
    }

    if (fs::is_regular_file(inputPath)) {
        cv::Mat       cvimage = cv::imread(inputPath, cv::IMREAD_COLOR);
        // cv::cvtColor(cvimage, cvimage, cv::COLOR_BGR2RGB);  // It is better to use RGB images, but the impact of using BGR on the results is not significant.
        deploy::Image image(cvimage.data, cvimage.cols, cvimage.rows);
        auto          result = model->predict(image);
        if (!outputPath.empty()) {
            cv::cvtColor(cvimage, cvimage, cv::COLOR_RGB2BGR);
            visualize(cvimage, result, labels);
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
                    visualize(images[j], results[j], labels);
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