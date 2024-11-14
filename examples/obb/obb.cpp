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

// Converts a bounding box with a given angle to its four corner points
std::vector<cv::Point> xyxyr2xyxyxyxy(const deploy::RotatedBox& box) {
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

// Visualize inference results
void visualize(cv::Mat& image, const deploy::OBBResult& result, const std::vector<std::pair<std::string, cv::Scalar>>& labelColorPairs) {
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
        auto     corners   = xyxyr2xyxyxyxy(box);
        cv::polylines(image, {corners}, true, color, 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(corners[0].x, corners[0].y - labelSize.height), cv::Point(corners[0].x + labelSize.width, corners[0].y), color, -1);
        cv::putText(image, labelText, corners[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    }
}

// Create model
std::shared_ptr<deploy::BaseOBB> createModel(const std::string& enginePath, bool useCudaGraph) {
    if (useCudaGraph) {
        return std::make_shared<deploy::DeployCGOBB>(enginePath);
    } else {
        return std::make_shared<deploy::DeployOBB>(enginePath);
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