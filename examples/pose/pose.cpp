#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <random>

#include "deploy/utils/utils.hpp"
#include "deploy/vision/inference.hpp"
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

// Visualize inference results
void visualize(cv::Mat& image, deploy::PoseResult& result, std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs) {
    std::vector<std::pair<int, int>> skeleton = {
        {16, 14},
        {14, 12},
        {17, 15},
        {15, 13},
        {12, 13},
        { 6, 12},
        { 7, 13},
        { 6,  7},
        { 6,  8},
        { 7,  9},
        { 8, 10},
        { 9, 11},
        { 2,  3},
        { 1,  2},
        { 1,  3},
        { 2,  4},
        { 3,  5},
        { 4,  6},
        { 5,  7}
    };

    for (size_t i = 0; i < result.num; ++i) {
        auto&       box       = result.boxes[i];
        int         cls       = result.classes[i];
        float       score     = result.scores[i];
        auto&       label     = labelColorPairs[cls].first;
        auto&       color     = labelColorPairs[cls].second;
        std::string labelText = label + " " + cv::format("%.2f", score);

        // Draw rectangle and label
        int      baseLine;
        cv::Size labelSize = cv::getTextSize(labelText, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), color, 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - labelSize.height), cv::Point(box.left + labelSize.width, box.top), color, -1);
        cv::putText(image, labelText, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

        int  nkpt = result.kpts[i].size();
        bool pose = nkpt == 17;
        for (size_t j = 0; j < nkpt; ++j) {
            if (int(result.kpts[i][j].x) % image.cols != 0 && int(result.kpts[i][j].y) % image.rows != 0) {
                if (result.kpts[i][j].conf.has_value() && result.kpts[i][j].conf.value() < 0.25) {
                    continue;
                }
                cv::circle(image, cv::Point(result.kpts[i][j].x, result.kpts[i][j].y), 3, color, -1, cv::LINE_AA);
            }
        }
        if (pose) {
            for (const auto& sk : skeleton) {
                const auto& kpt1 = result.kpts[i][sk.first - 1];
                const auto& kpt2 = result.kpts[i][sk.second - 1];

                if (kpt1.conf < 0.25 || kpt2.conf < 0.25) {
                    continue;
                }
                if (int(kpt1.x) % image.cols == 0 || int(kpt1.y) % image.rows == 0 || int(kpt1.x) < 0 || kpt1.y < 0) {
                    continue;
                }
                if (int(kpt2.x) % image.cols == 0 || int(kpt2.y) % image.rows == 0 || int(kpt2.x) < 0 || kpt2.y < 0) {
                    continue;
                }
                cv::line(image, cv::Point(kpt1.x, kpt1.y), cv::Point(kpt2.x, kpt2.y), color, 2, cv::LINE_AA);
            }
        }
    }
}

// Create model
std::shared_ptr<deploy::BasePose> createModel(const std::string& enginePath, bool useCudaGraph) {
    if (useCudaGraph) {
        return std::make_shared<deploy::DeployCGPose>(enginePath);
    } else {
        return std::make_shared<deploy::DeployPose>(enginePath);
    }
}

// Parse arguments
void parseArguments(int argc, char** argv, std::string& enginePath, std::string& inputPath, std::string& outputPath, std::string& labelPath, bool& useCudaGraph) {
    // Using a library for argument parsing (e.g., Boost.Program_options)
    // For simplicity, manual parsing is shown here
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " -e <engine> -i <input> [-o <output>] [-l <labels>] [--cudaGraph]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-e" || arg == "--engine") {
            enginePath = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            inputPath = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            outputPath = argv[++i];
        } else if (arg == "-l" || arg == "--labels") {
            labelPath = argv[++i];
        } else if (arg == "--cudaGraph") {
            useCudaGraph = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char** argv) {
    try {
        std::string enginePath, inputPath, outputPath, labelPath;
        bool        useCudaGraph = false;

        parseArguments(argc, argv, enginePath, inputPath, outputPath, labelPath, useCudaGraph);

        if (!fs::exists(enginePath)) {
            throw std::runtime_error("Engine path does not exist: " + enginePath);
        }
        if (!fs::exists(inputPath) || (!fs::is_regular_file(inputPath) && !fs::is_directory(inputPath))) {
            throw std::runtime_error("Input path does not exist or is not a regular file/directory: " + inputPath);
        }

        std::vector<std::pair<std::string, cv::Scalar>> labels;
        if (!outputPath.empty()) {
            if (labelPath.empty()) {
                throw std::runtime_error("Please provide a labels file using -l or --labels.");
            }
            if (!fs::exists(labelPath)) {
                throw std::runtime_error("Label path does not exist: " + labelPath);
            }

            labels = generateLabelColorPairs(labelPath);
            createOutputDirectory(outputPath);
        }

        auto model = createModel(enginePath, useCudaGraph);

        if (fs::is_regular_file(inputPath)) {
            cv::Mat cvimage = cv::imread(inputPath, cv::IMREAD_COLOR);
            if (cvimage.empty()) {
                throw std::runtime_error("Failed to read image from path: " + inputPath);
            }
            cv::cvtColor(cvimage, cvimage, cv::COLOR_BGR2RGB);
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
                    if (image.empty()) {
                        throw std::runtime_error("Failed to read image from path: " + imageFiles[j]);
                    }
                    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
                    images.emplace_back(image);
                    imgBatch.emplace_back(image.data, image.cols, image.rows);
                    imgNameBatch.emplace_back(fs::path(imageFiles[j]).filename().string());
                }

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
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}