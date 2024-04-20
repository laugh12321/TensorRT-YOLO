#include <CLI/CLI.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "deploy/utils/utils.hpp"
#include "deploy/vision/detection.hpp"
#include "deploy/vision/visualize.hpp"

namespace fs = std::filesystem;

std::vector<std::string> GetImagesInDirectory(const std::string& folder_path) {
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (fs::is_regular_file(entry)) {
            const auto extension = entry.path().extension().string();
            if (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp") {
                image_files.push_back(entry.path().string());
            }
        }
    }
    return image_files;
}

std::string GetFileName(const std::string& file_path) {
    return fs::path(file_path).filename().string();
}

void CreateOutputDirectory(const std::string& output) {
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

void ProcessSingleImage(const std::string& image_path, const std::shared_ptr<deploy::DeployDet>& model, const std::vector<std::pair<std::string, cv::Scalar>>& labels, const std::string& output_path) {
    cv::Mat image  = cv::imread(image_path);
    auto    result = model->Predict(image);
    if (!output_path.empty()) {
        deploy::Visualize(image, result, labels);
        cv::imwrite(output_path + "/" + GetFileName(image_path), image);
    }
}

void ProcessBatchImages(const std::vector<std::string>& image_files, const std::shared_ptr<deploy::DeployDet>& model, const std::vector<std::pair<std::string, cv::Scalar>>& labels, const std::string& output_path) {
    const size_t                                         batch_size = model->batch_size;
    deploy::GpuTimer                                     gpu_timer;
    deploy::CpuTimer<std::chrono::high_resolution_clock> cpu_timer;
    for (size_t i = 0; i < image_files.size(); i += batch_size) {
        std::vector<cv::Mat>     img_batch;
        std::vector<std::string> img_name_batch;
        for (size_t j = i; j < i + batch_size && j < image_files.size(); ++j) {
            img_name_batch.push_back(GetFileName(image_files[j]));
            img_batch.push_back(cv::imread(image_files[j]));
        }
        cpu_timer.Start();
        gpu_timer.Start();
        auto results = model->Predict(img_batch);
        gpu_timer.Stop();
        cpu_timer.Stop();
        std::cout << img_batch.size() << " Batch infer CPU elapsed time: " << cpu_timer.Microseconds() / 1000 << " ms" << std::endl;
        std::cout << img_batch.size() << " Batch infer GPU elapsed time: " << gpu_timer.Microseconds() / 1000 << " ms" << std::endl;
        gpu_timer.Reset();
        cpu_timer.Reset();
        if (!output_path.empty()) {
            for (size_t j = 0; j < img_batch.size(); ++j) {
                deploy::Visualize(img_batch[j], results[j], labels);
                cv::imwrite(output_path + "/" + img_name_batch[j], img_batch[j]);
            }
        }
    }
}

int main(int argc, char** argv) {
    CLI::App app{"YOLO Series Inference Script."};

    std::string engine_path;
    std::string input_path;
    std::string output_path;
    std::string label_path;
    app.add_option("-e, --engine", engine_path, "Serialized TensorRT engine.")
        ->required()
        ->check(CLI::ExistingFile);
    app.add_option("-i, --input", input_path, "Path to image or directory.")
        ->required()
        ->check(CLI::ExistingPath);
    app.add_option("-o, --output", output_path, "Directory to save results.");
    app.add_option("-l, --labels", label_path, "File to use for reading the class labels from")
        ->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    auto model = std::make_shared<deploy::DeployDet>(engine_path);

    std::vector<std::pair<std::string, cv::Scalar>> labels;
    if (!output_path.empty()) {
        labels = deploy::GenerateLabelColorParis(label_path);
        CreateOutputDirectory(output_path);
    }

    if (fs::is_regular_file(input_path)) {
        ProcessSingleImage(input_path, model, labels, output_path);
    } else {
        auto image_files = GetImagesInDirectory(input_path);
        if (!image_files.empty()) {
            ProcessBatchImages(image_files, model, labels, output_path);
        } else {
            std::cerr << "No images found in directory: " << input_path << std::endl;
            return 1;
        }
    }

    std::cout << "Inference completed." << std::endl;
    return 0;
}
