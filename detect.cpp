#include "include/common.hpp"
#include "include/cuda_utils.hpp"
#include "include/detection.hpp"

#include <CLI/CLI.hpp>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::string> getImagesInDirectory(const std::string& folderPath) {
    std::vector<std::string> imageFiles;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (fs::is_regular_file(entry)) {
            const auto extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
                imageFiles.push_back(entry.path().string());
            }
        }
    }

    return imageFiles;
}

std::string getFileName(const std::string& filePath) {
    return fs::path(filePath).filename().string();
}

void createOutputDirectory(const std::string& output) {
    if (!fs::exists(output)) {
        if (!fs::create_directories(output)) {
            std::cerr << "Failed to create output directory: " << output << std::endl;
            exit(1);
        } else {
            std::cout << "Output directory created: " << output << std::endl;
        }
    } else if (!fs::is_directory(output)) {
        std::cerr << "Output path exists but is not a directory: " << output << std::endl;
        exit(1);
    }
}

int main(int argc, char** argv) {
    CLI::App app{"YOLO Series Inference Script."};

    std::string enginePath, inputPath, outputPath, labelPath;
    app.add_option("-e, --engine", enginePath, "Serialized TensorRT engine.")->required()->check(CLI::ExistingFile);
    app.add_option("-i, --input", inputPath, "Path to image or directory.")->required()->check(CLI::ExistingPath);
    app.add_option("-o, --output", outputPath, "Directory to save results.");
	app.add_option("-l, --labels", labelPath, "File to use for reading the class labels from")->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    auto engine = yolo::load(enginePath);
    if (!engine) throw std::runtime_error("Failed to construct engine context.");
    std::cout << "Engine context constructed successfully." << std::endl;

    std::vector<std::pair<std::string, cv::Scalar>> labelColors;
    if (!outputPath.empty()) {
        labelColors = generateLabelsWithColors(labelPath);
        createOutputDirectory(outputPath);
    }

    if (fs::is_regular_file(inputPath)) {
        cv::Mat image = cv::imread(inputPath);
        auto detectInfo = engine->predict(image);

        if (!outputPath.empty()) {
            visualize(image, detectInfo, labelColors);
            cv::imwrite(outputPath + "/" + getFileName(inputPath), image);
        }
    } else {
        cuda_utils::GpuTimer gpuTimer;
        cuda_utils::CpuTimer<std::chrono::high_resolution_clock> cpuTimer;
		std::vector<std::pair<std::string, cv::Scalar>> labelColors;

        auto imageFiles = getImagesInDirectory(inputPath);
        const size_t batchSize = engine->batchSize;
        for (size_t i = 0; i < imageFiles.size(); i += batchSize) {
            std::vector<cv::Mat> imgBatch;
            std::vector<std::string> imgNameBatch;
            for (size_t j = i; j < i + batchSize && j < imageFiles.size(); ++j) {
                imgNameBatch.push_back(getFileName(imageFiles[j]));
                imgBatch.push_back(cv::imread(imageFiles[j]));
            }

            cpuTimer.start();
            gpuTimer.start();
            auto detectInfos = engine->predict(imgBatch);
            gpuTimer.stop();
            cpuTimer.stop();

            std::cout << imgBatch.size() << " Batch infer CPU elapsed time: " << cpuTimer.milliseconds() << " ms" << std::endl;
            std::cout << imgBatch.size() << " Batch infer GPU elapsed time: " << gpuTimer.milliseconds() << " ms" << std::endl;
            gpuTimer.reset();
            cpuTimer.reset();

            if (!outputPath.empty()) {
                for (size_t j = 0; j < imgBatch.size(); ++j) {
                    visualize(imgBatch[j], detectInfos[j], labelColors);
                    cv::imwrite(outputPath + "/" + imgNameBatch[j], imgBatch[j]);
                }
            }
        }
    }

	std::cout << "Inference completed." << std::endl;
    return 0;
}
