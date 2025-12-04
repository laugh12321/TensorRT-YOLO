/**
 * @file segment.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief Segment C++ 示例
 * @date 2025-06-07
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>

#include "trtyolo.hpp"

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

// 可视化推理结果（分割任务）
void visualize(cv::Mat& image, trtyolo::SegmentRes& result, const std::vector<std::string>& labels) {
    int im_h = image.rows;  // 图像高度
    int im_w = image.cols;  // 图像宽度

    // 遍历每个检测到的目标
    for (size_t i = 0; i < result.num; ++i) {
        auto&       box        = result.boxes[i];                          // 当前目标的边界框
        int         cls        = result.classes[i];                        // 当前目标的类别
        float       score      = result.scores[i];                         // 当前目标的置信度
        auto&       label      = labels[cls];                              // 获取类别对应的标签
        std::string label_text = label + " " + cv::format("%.3f", score);  // 构造显示的标签文本

        // 绘制边界框和标签
        int      base_line;
        cv::Size label_size = cv::getTextSize(label_text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &base_line);
        cv::rectangle(image, cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(251, 81, 163), 2, cv::LINE_AA);
        cv::rectangle(image, cv::Point(box.left, box.top - label_size.height), cv::Point(box.left + label_size.width, box.top), cv::Scalar(125, 40, 81), -1);
        cv::putText(image, label_text, cv::Point(box.left, box.top), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(253, 168, 208), 1);

        // 创建分割掩码
        auto xyxy = box.xyxy();
        int  w    = std::max(xyxy[2] - xyxy[0] + 1, 1);
        int  h    = std::max(xyxy[3] - xyxy[1] + 1, 1);

        int x1 = std::max(0, xyxy[0]);
        int y1 = std::max(0, xyxy[1]);
        int x2 = std::min(im_w, xyxy[2] + 1);
        int y2 = std::min(im_h, xyxy[3] + 1);

        // 将模型输出的浮点掩码转换为 OpenCV 的 Mat 格式
        cv::Mat float_mask(result.masks[i].height, result.masks[i].width, CV_32FC1, result.masks[i].data.data());
        cv::resize(float_mask, float_mask, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

        // 将浮点掩码转换为布尔掩码
        cv::Mat bool_mask;
        cv::threshold(float_mask, bool_mask, 0.5, 255, cv::THRESH_BINARY);

        // 创建一个与原图大小相同的掩码图像
        cv::Mat mask_image = cv::Mat::zeros(image.size(), CV_8UC1);

        // 计算从 bool_mask 中裁切的偏移（若 xyxy 左上角为负，则偏移为负值的绝对值）
        int src_x_offset = std::max(0, -xyxy[0]);
        int src_y_offset = std::max(0, -xyxy[1]);

        // 目标区域的宽高（在原图中的实际可用区域）
        int target_w = x2 - x1;
        int target_h = y2 - y1;
        if (target_w <= 0 || target_h <= 0)
            continue;

        // 确保源裁切区域不会超出 bool_mask 的边界
        if (src_x_offset + target_w > bool_mask.cols)
            target_w = bool_mask.cols - src_x_offset;
        if (src_y_offset + target_h > bool_mask.rows)
            target_h = bool_mask.rows - src_y_offset;
        if (target_w <= 0 || target_h <= 0)
            continue;

        // 定义目标区域和源区域
        cv::Rect target_rect(x1, y1, target_w, target_h);                      // 目标区域在 mask_image 中的位置
        cv::Rect source_rect(src_x_offset, src_y_offset, target_w, target_h);  // 源区域在 bool_mask 中的位置

        // 将 bool_mask 的指定区域复制到 mask_image 的指定区域
        bool_mask(source_rect).copyTo(mask_image(target_rect));

        // 创建一个与原图大小相同的颜色图像
        cv::Mat color_image(image.size(), image.type(), cv::Scalar(251, 81, 163));

        // 使用掩码将颜色图像与原图进行混合
        cv::Mat masked_color_image;
        cv::bitwise_and(color_image, color_image, masked_color_image, mask_image);

        cv::addWeighted(image, 1.0, masked_color_image, 0.5, 0, image);
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
void process_single_image(const std::string& image_path, const std::string& output_path, trtyolo::SegmentModel& model, const std::vector<std::string>& labels) {
    cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to read image from path: " + image_path);
    }

    trtyolo::Image img(image.data, image.cols, image.rows);
    auto           result = model.predict(img);

    if (!output_path.empty()) {
        visualize(image, result, labels);
        fs::path output_file_path = output_path / fs::path(image_path).filename();
        cv::imwrite(output_file_path.string(), image);
    }
}

// 处理一批图像
void process_batch_images(const std::vector<std::string>& image_paths, const std::string& output_path, trtyolo::SegmentModel& model, const std::vector<std::string>& labels) {
    const int batch_size = model.batch();
    for (size_t i = 0; i < image_paths.size(); i += batch_size) {
        std::vector<cv::Mat>        images;
        std::vector<trtyolo::Image> img_batch;
        std::vector<std::string>    img_name_batch;

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

        trtyolo::InferOption option;
        option.enableSwapRB();

        if (!fs::is_regular_file(input_path)) {
            option.enablePerformanceReport();
        }

        auto model = std::make_unique<trtyolo::SegmentModel>(engine_path, option);

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

        if (!fs::is_regular_file(input_path)) {
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
