#!/usr/bin/env python
# -*-coding:utf-8 -*-
# ==============================================================================
# Copyright (c) 2025 laugh12321 Authors. All Rights Reserved.
#
# Licensed under the GNU General Public License v3.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# File    :   mutli_thread_process.py
# Version :   6.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2025/01/21 17:43:59
# Desc    :   多线程示例
# ==============================================================================
import argparse
import time
from multiprocessing import Pool
from threading import Thread

import cv2
from tensorrt_yolo.infer import DetectModel, InferOption, image_batches


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-threaded and multi-process inference script.")
    parser.add_argument("--engine", required=True, help="Path to the TensorRT engine file.")
    parser.add_argument("--image_path", type=str, required=True, help="Directory or file list of images to predict.")
    parser.add_argument("--thread_num", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--use_multi_process", action="store_true", help="Whether to use multi-process.")
    parser.add_argument("--process_num", type=int, default=1, help="Number of processes to use.")
    return parser.parse_args()


def build_option():
    option = InferOption()
    option.enable_swap_rb()
    option.enable_performance_report()
    return option


def load_model(engine_file):
    """加载模型"""
    option = build_option()
    global model
    model = DetectModel(engine_file, option=option)


def process_predict(img_batch):
    """子进程中的预测函数"""
    # global model
    images = [cv2.imread(image_path) for image_path in img_batch]
    result = model.predict(images)


def predict(model, img_batches):
    """多线程中的预测函数"""
    for img_batch in img_batches:
        images = [cv2.imread(image_path) for image_path in img_batch]
        result = model.predict(images)
    model.performance_report()


def main():
    args = parse_arguments()
    img_batches = image_batches(args.image_path, 1, True)

    # 记录开始时间
    start_time = time.time()

    if args.use_multi_process:
        with Pool(args.process_num, initializer=load_model, initargs=(args.engine,)) as pool:
            pool.map(process_predict, img_batches)
    else:
        # 使用多线程
        load_model(args.engine)
        threads = []
        for i in range(args.thread_num):
            t = Thread(target=predict, args=(model.clone(), img_batches))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

    # 记录结束时间
    end_time = time.time()

    # 输出耗时
    print(f"Total time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
