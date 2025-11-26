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
# Version :   6.4.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2025/01/21 17:43:59
# Desc    :   多线程示例
# ==============================================================================
import argparse
import time
from multiprocessing import Pool
from pathlib import Path
from threading import Thread

from trtyolo import TRTYOLO


def parse_arguments():
    parser = argparse.ArgumentParser(description="Multi-threaded and multi-process inference script.")
    parser.add_argument("--engine", required=True, help="Path to the TensorRT engine file.")
    parser.add_argument("--image_path", type=str, required=True, help="Directory or file list of images to predict.")
    parser.add_argument("--thread_num", type=int, default=1, help="Number of threads to use.")
    parser.add_argument("--use_multi_process", action="store_true", help="Whether to use multi-process.")
    parser.add_argument("--process_num", type=int, default=1, help="Number of processes to use.")
    return parser.parse_args()


def load_model(engine_file):
    """加载模型"""
    model = TRTYOLO(engine_file, task="detect", swap_rb=True, profile=True)
    return model


def process_predict(img_batch, engine_file):
    """子进程中的预测函数"""
    model = load_model(engine_file)
    model.predict(img_batch)
    throughput, cpu_latency, gpu_latency = model.profile()
    print(throughput)
    print(cpu_latency)
    print(gpu_latency)


def predict(model, img_batch):
    """多线程中的预测函数"""
    model.predict(img_batch)
    throughput, cpu_latency, gpu_latency = model.profile()
    print(throughput)
    print(cpu_latency)
    print(gpu_latency)


def main():
    args = parse_arguments()
    image_files = [args.image_path] if Path(args.image_path).is_file() else list(Path(args.image_path).glob("**/*.jpg"))

    # 记录开始时间
    start_time = time.time()

    if args.use_multi_process:
        image_batches = [image_files[i : i + args.process_num] for i in range(0, len(image_files), args.process_num)]
        with Pool(args.process_num) as pool:
            pool.starmap(process_predict, [(img_batch, args.engine) for img_batch in image_batches])
    else:
        # 使用多线程
        model = load_model(args.engine)
        threads = []
        image_batches = [image_files[i : i + args.thread_num] for i in range(0, len(image_files), args.thread_num)]
        for i in range(args.thread_num):
            t = Thread(target=predict, args=(model.clone(), image_batches[i]))
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
