#!/usr/bin/env python
# -*-coding:utf-8 -*-
# ==============================================================================
# Copyright (c) 2024 laugh12321 Authors. All Rights Reserved.
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
# File    :   detect.py
# Version :   5.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/29 15:13:41
# Desc    :   YOLO Series Inference For Pose Estimation.
# ==============================================================================
import argparse
import sys
from pathlib import Path

import cv2
from loguru import logger
from rich.progress import track

from tensorrt_yolo.infer import CpuTimer, DeployCGPose, DeployPose, GpuTimer, generate_labels_with_colors, image_batches, visualize


def main():
    parser = argparse.ArgumentParser(description='YOLO Series Inference For Pose Estimation.')
    parser.add_argument('-e', '--engine', required=True, type=str, help='The serialized TensorRT engine.')
    parser.add_argument('-i', '--input', required=True, type=str, help="Path to the image or directory to process.")
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory where to save the visualization results.')
    parser.add_argument(
        "-l", "--labels", default="./labels.txt", help="File to use for reading the class labels from, default: ./labels.txt"
    )
    parser.add_argument(
        '--cudaGraph', action='store_true', help='Optimize inference using CUDA Graphs, compatible with static models only.'
    )

    args = parser.parse_args()

    if args.output and not args.labels:
        logger.error("Please provide a labels file using -l or --labels.")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        args.labels = generate_labels_with_colors(args.labels)

    model = DeployCGPose(args.engine) if args.cudaGraph else DeployPose(args.engine)

    batchs = image_batches(args.input, model.batch, args.cudaGraph)

    if len(batchs) > 2:
        cpu_timer = CpuTimer()
        gpu_timer = GpuTimer()

    logger.info(f"Infering data in {args.input}")
    for batch in track(batchs, description="[cyan]Processing batches", total=len(batchs)):
        images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in batch]

        if len(batchs) > 2:
            cpu_timer.start()
            gpu_timer.start()

        results = model.predict(images)

        if len(batchs) > 2:
            cpu_timer.stop()
            gpu_timer.stop()

        if args.output:
            for image_path, image, result in zip(batch, images, results):
                vis_image = visualize(image, result, args.labels)
                cv2.imwrite(str(output_dir / Path(image_path).name), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    logger.success("Finished Inference.")

    if len(batchs) > 2:
        logger.success(
            "Benchmark results include time for H2D and D2H memory copies, preprocessing, and postprocessing.\n"
            f"    CPU Average Latency: {cpu_timer.milliseconds() / len(batchs):.3f} ms\n"
            f"    GPU Average Latency: {gpu_timer.milliseconds() / len(batchs):.3f} ms\n"
            "    Finished Inference."
        )


if __name__ == '__main__':
    main()
