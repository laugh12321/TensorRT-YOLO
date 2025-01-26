/**
 * @file obb.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief OBB C++ 示例
 * @date 2025-01-23
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "deploy/model.hpp"
#include "deploy/option.hpp"
#include "deploy/result.hpp"

namespace fs = std::filesystem;

// 获取指定目录中的图像文件
std::vector<std::string> get_images_in_directory(const std::string& folder_path) {
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        const auto extension = entry.path().extension().string();
        if (fs::is_regular_file(entry) && (extension == ".jpg" || extension == ".png" || extension == ".jpeg" || extension == ".bmp")) {
            image_files.push_back(entry.path().string());
        }
    }
    return image_files;
}

// 创建输出目录
void create_output_directory(const std::string& output_path) {
    if (!fs::exists(output_path) && !fs::create_directories(output_path)) {
        throw std::runtime_error("Failed to create output directory: " + output_path);
    } else if (!fs::is_directory(output_path)) {
        throw std::runtime_error("Output path exists but is not a directory: " + output_path);
    }
}

// 从文件中生成标签
std::vector<std::string> generate_labels(const std::string& label_file) {
    std::ifstream file(label_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open labels file: " + label_file);
    }

    std::vector<std::string> labels;
    std::string              label;
    while (std::getline(file, label)) {
        labels.emplace_back(label);
    }
    return labels;
}

// 将带角度的边界框转换为四个角点
std::vector<cv::Point> xyxyr_to_xyxyxyxy(const deploy::RotatedBox& box) {
    // 计算角度的余弦和正弦值
    float cos_value = std::cos(box.theta);
    float sin_value = std::sin(box.theta);

    // 计算边界框的中心坐标
    float center_x = (box.left + box.right) * 0.5f;
    float center_y = (box.top + box.bottom) * 0.5f;

    // 计算边界框的半宽和半高
    float half_width  = (box.right - box.left) * 0.5f;
    float half_height = (box.bottom - box.top) * 0.5f;

    // 计算旋转后的角点向量
    float vec_x1 = half_width * cos_value;
    float vec_y1 = half_width * sin_value;
    float vec_x2 = half_height * sin_value;
    float vec_y2 = half_height * cos_value;

    // 返回旋转矩形的四个角点
    return {
        cv::Point(center_x + vec_x1 - vec_x2, center_y + vec_y1 + vec_y2),
        cv::Point(center_x + vec_x1 + vec_x2, center_y + vec_y1 - vec_y2),
        cv::Point(center_x - vec_x1 + vec_x2, center_y - vec_y1 - vec_y2),
        cv::Point(center_x - vec_x1 - vec_x2, center_y - vec_y1 + vec_y2)};
}

// 可视化推理结果
void visualize(cv::Mat& image, deploy::OBBRes& result, const std::vector<std::string>& labels) {
    for (size_t i = 0; i < result.num; ++i) {
        auto&       box        = result.boxes[i];                          // 当前边界框
        int         cls        = result.classes[i];                        // 当前类别
        float       score      = result.scores[i];                         // 当前置信度
        auto&       label      = labels[cls];                              // 获取类别标签
        std::string label_text = label + " " + cv::format("%.3f", score);  // 构造显示的标签文本

        // 计算标签的尺寸
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);

        // 获取旋转矩形的四个角点
        auto corners = xyxyr_to_xyxyxyxy(box);

        // 绘制旋转矩形
        cv::polylines(image, {corners}, true, cv::Scalar(251, 81, 163), 2, cv::LINE_AA);

        // 绘制标签背景
        cv::rectangle(image, cv::Point(corners[0].x, corners[0].y - label_size.height),
                      cv::Point(corners[0].x + label_size.width, corners[0].y),
                      cv::Scalar(125, 40, 81), -1);

        // 绘制标签文本
        cv::putText(image, label_text, corners[0], cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);
    }
}

// 解析命令行参数
void parse_arguments(int argc, char** argv, std::string& engine_path, std::string& input_path, std::string& output_path, std::string& label_path) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " -e <engine> -i <input> [-o <output>] [-l <labels>]" << std::endl;
        std::exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-e" || arg == "--engine") {
            engine_path = argv[++i];
        } else if (arg == "-i" || arg == "--input") {
            input_path = argv[++i];
        } else if (arg == "-o" || arg == "--output") {
            output_path = argv[++i];
        } else if (arg == "-l" || arg == "--labels") {
            label_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
}

// 处理单张图像
void process_single_image(const std::string& image_path, const std::string& output_path, deploy::OBBModel& model, const std::vector<std::string>& labels) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image from path: " + image_path);
    }

    deploy::Image img(image.data, image.cols, image.rows);
    auto          result = model.predict(img);

    if (!output_path.empty()) {
        visualize(image, result, labels);
        fs::path output_file_path = output_path / fs::path(image_path).filename();
        cv::imwrite(output_file_path.string(), image);
    }
}

// 处理一批图像
void process_batch_images(const std::vector<std::string>& image_paths, const std::string& output_path, deploy::OBBModel& model, const std::vector<std::string>& labels) {
    const int batch_size = model.batch_size();
    for (size_t i = 0; i < image_paths.size(); i += batch_size) {
        std::vector<cv::Mat>       images;
        std::vector<deploy::Image> img_batch;
        std::vector<std::string>   img_name_batch;

        for (size_t j = i; j < i + batch_size && j < image_paths.size(); ++j) {
            cv::Mat image = cv::imread(image_paths[j], cv::IMREAD_COLOR);
            if (image.empty()) {
                throw std::runtime_error("Failed to read image from path: " + image_paths[j]);
            }
            images.push_back(image);
            img_batch.emplace_back(image.data, image.cols, image.rows);
            img_name_batch.push_back(fs::path(image_paths[j]).filename().string());
        }

        auto results = model.predict(img_batch);

        if (!output_path.empty()) {
            for (size_t j = 0; j < images.size(); ++j) {
                visualize(images[j], results[j], labels);
                fs::path output_file_path = output_path + "/" + img_name_batch[j];
                cv::imwrite(output_file_path.string(), images[j]);
            }
        }
    }
}

int main(int argc, char** argv) {
    try {
        std::string engine_path, input_path, output_path, label_path;
        parse_arguments(argc, argv, engine_path, input_path, output_path, label_path);

        if (!fs::exists(engine_path)) {
            throw std::runtime_error("Engine path does not exist: " + engine_path);
        }
        if (!fs::exists(input_path) || (!fs::is_regular_file(input_path) && !fs::is_directory(input_path))) {
            throw std::runtime_error("Input path does not exist or is not a regular file/directory: " + input_path);
        }

        std::vector<std::string> labels;
        if (!output_path.empty()) {
            if (label_path.empty()) {
                throw std::runtime_error("Please provide a labels file using -l or --labels.");
            }
            if (!fs::exists(label_path)) {
                throw std::runtime_error("Label path does not exist: " + label_path);
            }
            labels = generate_labels(label_path);
            create_output_directory(output_path);
        }

        deploy::InferOption option;
        option.enableSwapRB();

        if (!fs::is_regular_file(input_path)) {
            option.enablePerformanceReport();
        }

        auto model = std::make_unique<deploy::OBBModel>(engine_path, option);

        if (fs::is_regular_file(input_path)) {
            process_single_image(input_path, output_path, *model, labels);
        } else {
            auto image_files = get_images_in_directory(input_path);
            if (image_files.empty()) {
                throw std::runtime_error("Failed to read image from path: " + input_path);
            }
            process_batch_images(image_files, output_path, *model, labels);
        }

        std::cout << "Inference completed." << std::endl;

        if (option.enable_performance_report) {
            auto [throughput_str, gpu_latency_str, cpu_latency_str] = model->performanceReport();
            std::cout << throughput_str << std::endl;
            std::cout << gpu_latency_str << std::endl;
            std::cout << cpu_latency_str << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return 0;
}
