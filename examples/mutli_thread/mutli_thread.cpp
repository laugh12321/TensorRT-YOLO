/**
 * @file mutli_thread.cpp
 * @author laugh12321 (laugh12321@vip.qq.com)
 * @brief 多线程示例
 * @date 2025-06-07
 *
 * @copyright Copyright (c) 2025 laugh12321. All Rights Reserved.
 *
 */

#include <algorithm>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>
#include <vector>

#include "trtyolo.hpp"

std::vector<std::vector<std::string>> get_images_list(const std::string& image_file_path, int thread_num) {
    std::vector<cv::String> images;
    cv::glob(image_file_path, images, false);

    // 过滤图片，只保留特定扩展名的图片
    std::vector<cv::String> filtered_images;
    for (const auto& image_path : images) {
        std::string extension = image_path.substr(image_path.find_last_of(".") + 1);
        if (extension == "jpg" || extension == "png" || extension == "jpeg" || extension == "bmp") {
            filtered_images.push_back(image_path);
        }
    }

    size_t count          = filtered_images.size();
    size_t num_per_thread = count / thread_num;  // 每个线程分配的图像数量
    size_t remaining      = count % thread_num;  // 剩余的图像数量

    std::vector<std::vector<std::string>> image_list(thread_num);

    for (int i = 0; i < thread_num; ++i) {
        // 显式将 i 转换为 size_t 类型
        size_t start_idx = i * num_per_thread + std::min(static_cast<size_t>(i), remaining);
        size_t end_idx   = start_idx + num_per_thread + (i < remaining ? 1 : 0);

        // 将图像路径分配到对应的线程列表
        for (size_t j = start_idx; j < end_idx; ++j) {
            image_list[i].push_back(filtered_images[j]);
        }
    }

    return image_list;
}

void predict(trtyolo::DetectModel* model, int thread_id, const std::vector<std::string>& image_paths) {
    auto batch_size = model->batch();

    for (size_t start_idx = 0; start_idx < image_paths.size(); start_idx += batch_size) {
        std::vector<trtyolo::Image> img_batch;

        std::transform(image_paths.begin() + start_idx,
                       image_paths.begin() + std::min(start_idx + batch_size, image_paths.size()),
                       std::back_inserter(img_batch), [](const std::string& image_path) {
                           cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
                           if (image.empty()) {
                               throw std::runtime_error("Failed to read image from path: " + image_path);
                           }
                           return trtyolo::Image{image.data, image.cols, image.rows};
                       });

        auto results = model->predict(img_batch);
    }

    auto [throughput_str, gpu_latency_str, cpu_latency_str] = model->performanceReport();

    std::cout << "Thread Id: " << thread_id << ", Throughput: " << throughput_str << std::endl;
    std::cout << "Thread Id: " << thread_id << ", GPU Latency: " << gpu_latency_str << std::endl;
    std::cout << "Thread Id: " << thread_id << ", CPU Latency: " << cpu_latency_str << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <image_dir> <thread_num>" << std::endl;
        std::cerr << "e.g: ./multi_thread_demo ./yolo11n.engine ./images 3" << std::endl;
        return -1;
    }

    // 解析命令行参数
    std::string model_path           = argv[1];
    std::string image_dir            = argv[2];
    int         requested_thread_num = std::atoi(argv[3]);  // 用户请求的线程数

    // 获取系统支持的最大线程数
    int max_thread_num = std::thread::hardware_concurrency();
    if (max_thread_num <= 0) {
        max_thread_num = 2;  // 如果无法获取硬件线程数，设置默认值
    }

    // 限制线程数不超过系统支持的最大线程数
    int thread_num = std::min(requested_thread_num, max_thread_num);
    std::cout << "Using " << thread_num << " threads (max supported: " << max_thread_num << ")." << std::endl;

    // 初始化模型选项
    auto option = trtyolo::InferOption();
    option.enableSwapRB();
    option.enablePerformanceReport();

    // 加载模型
    auto model = trtyolo::DetectModel(model_path, option);

    // 克隆模型到多个线程
    std::vector<std::unique_ptr<trtyolo::DetectModel>> models;
    models.reserve(thread_num);
    for (int i = 0; i < thread_num; ++i) {
        models.emplace_back(model.clone());
    }

    // 获取图像列表
    auto image_files = get_images_list(image_dir, thread_num);

    // 创建线程池
    std::vector<std::thread> threads;
    threads.reserve(thread_num);

    for (int i = 0; i < thread_num; ++i) {
        threads.emplace_back(predict, models[i].get(), i, image_files[i]);
    }

    // 等待所有线程完成
    for (auto& thread : threads) {
        thread.join();
    }

    std::cout << "All threads completed successfully." << std::endl;
    return 0;
}