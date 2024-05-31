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

// Get file name from file path
std::string getFileName(const std::string& filePath) {
    return fs::path(filePath).filename().string();
}

// Create output directory
void createOutputDirectory(const std::string& outputPath) {
    if (!fs::exists(outputPath) && !fs::create_directories(outputPath)) {
        std::cerr << "Failed to create output directory: " << outputPath << std::endl;
        exit(1);
    } else if (!fs::is_directory(outputPath)) {
        std::cerr << "Output path exists but is not a directory: " << outputPath << std::endl;
        exit(1);
    }
}

// Generate label and color pairs
std::vector<std::pair<std::string, cv::Scalar>> generateLabelColorPairs(const std::string& labelFile) {
    std::vector<std::pair<std::string, cv::Scalar>> labelColorPairs;
    std::ifstream                                   file(labelFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open labels file: " << labelFile << std::endl;
        return labelColorPairs;
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
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
        int      baseLine;
        cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(image, cv::Point(box.left, box.top - labelSize.height), cv::Point(box.left + labelSize.width, box.top), color, -1);
        cv::putText(image, labelText, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
}

// Process a single image
void processSingleImage(const std::string& imagePath, const std::shared_ptr<deploy::DeployDet>& model, const std::string& outputPath, const std::vector<std::pair<std::string, cv::Scalar>>& labels) {
    cv::Mat cvimage = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (cvimage.empty()) {
        std::cerr << "Failed to read image: " << imagePath << std::endl;
        return;
    }
    // cv::cvtColor(cvimage, cvimage, cv::COLOR_BGR2RGB);  // It is better to use RGB images, but the impact of using BGR on the results is not significant.
    deploy::Image image(cvimage.data, cvimage.cols, cvimage.rows);
    auto          result = model->predict(image);
    if (!outputPath.empty()) {
        // cv::cvtColor(cvimage, cvimage, cv::COLOR_RGB2BGR);
        visualize(cvimage, result, labels);
        cv::imwrite(outputPath + "/" + getFileName(imagePath), cvimage);
    }
}

// Process a batch of images
void processBatchImages(const std::vector<std::string>& imageFiles, const std::shared_ptr<deploy::DeployDet>& model, const std::string& outputPath, const std::vector<std::pair<std::string, cv::Scalar>>& labels) {
    const size_t                                         batchSize = model->batch;
    deploy::GpuTimer                                     gpuTimer;
    deploy::CpuTimer<std::chrono::high_resolution_clock> cpuTimer;
    int                                                  count = 0;

    for (size_t i = 0; i < imageFiles.size(); i += batchSize) {
        std::vector<cv::Mat>     images;
        std::vector<std::string> imgNameBatch;

        for (size_t j = i; j < i + batchSize && j < imageFiles.size(); ++j) {
            cv::Mat image = cv::imread(imageFiles[j], cv::IMREAD_COLOR);
            if (image.empty()) {
                std::cerr << "Failed to read image: " << imageFiles[j] << std::endl;
                continue;
            }
            // cv::cvtColor(image, image, cv::COLOR_BGR2RGB); // It is better to use RGB images, but the impact of using BGR on the results is not significant.
            images.push_back(image);
            imgNameBatch.push_back(getFileName(imageFiles[j]));
        }

        if (images.empty()) continue;

        std::vector<deploy::Image> imgBatch;
        for (const auto& image : images) {
            imgBatch.emplace_back(image.data, image.cols, image.rows);
        }

        if (i > 5) {
            cpuTimer.start();
            gpuTimer.start();
        }

        auto results = model->predict(imgBatch);

        if (i > 5) {
            cpuTimer.stop();
            gpuTimer.stop();
            count++;
        }

        if (!outputPath.empty()) {
            for (size_t j = 0; j < images.size(); ++j) {
                // cv::cvtColor(images[j], images[j], cv::COLOR_RGB2BGR);
                visualize(images[j], results[j], labels);
                cv::imwrite(outputPath + "/" + imgNameBatch[j], images[j]);
            }
        }
    }

    if (count > 0) {
        std::cout << "Average infer CPU elapsed time: " << cpuTimer.microseconds() / 1000 / count << " ms" << std::endl;
        std::cout << "Average infer GPU elapsed time: " << gpuTimer.microseconds() / 1000 / count << " ms" << std::endl;
    }
}

int main(int argc, char** argv) {
    CLI::App app{"YOLO Series Inference Script"};

    std::string enginePath, inputPath, outputPath, labelPath;
    app.add_option("-e,--engine", enginePath, "Serialized TensorRT engine")->required()->check(CLI::ExistingFile);
    app.add_option("-i,--input", inputPath, "Path to image or directory")->required()->check(CLI::ExistingPath);
    app.add_option("-o,--output", outputPath, "Directory to save results");
    app.add_option("-l,--labels", labelPath, "File to use for reading the class labels from")->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    auto model = std::make_shared<deploy::DeployDet>(enginePath);

    std::vector<std::pair<std::string, cv::Scalar>> labels;
    if (!outputPath.empty()) {
        labels = generateLabelColorPairs(labelPath);
        createOutputDirectory(outputPath);
    }

    if (fs::is_regular_file(inputPath)) {
        processSingleImage(inputPath, model, outputPath, labels);
    } else {
        auto imageFiles = getImagesInDirectory(inputPath);
        if (!imageFiles.empty()) {
            processBatchImages(imageFiles, model, outputPath, labels);
        } else {
            std::cerr << "No images found in directory: " << inputPath << std::endl;
            return 1;
        }
    }

    std::cout << "Inference completed." << std::endl;
    return 0;
}
