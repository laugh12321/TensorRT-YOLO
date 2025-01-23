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
# File    :   detect.py
# Version :   6.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2025/01/23 14:16:11
# Desc    :   Detect 示例
# ==============================================================================
import argparse
import sys
from pathlib import Path

import cv2
from loguru import logger
from rich.progress import track
from tensorrt_yolo.infer import DetectModel, InferOption, generate_labels, image_batches, visualize


def main():
    parser = argparse.ArgumentParser(description='YOLO Series Inference For Detection.')
    parser.add_argument('-e', '--engine', required=True, type=str, help='The serialized TensorRT engine.')
    parser.add_argument('-i', '--input', required=True, type=str, help="Path to the image or directory to process.")
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory where to save the visualization results.')
    parser.add_argument(
        "-l", "--labels", default="./labels.txt", help="File to use for reading the class labels from, default: ./labels.txt"
    )

    args = parser.parse_args()

    if args.output and not args.labels:
        logger.error("Please provide a labels file using -l or --labels.")
        sys.exit(1)

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        args.labels = generate_labels(args.labels)

    option = InferOption()
    option.enable_swap_rb()
    option.enable_performance_report()
    # option.set_normalize_params([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # PP-YOLOE, PP-YOLOE+

    model = DetectModel(args.engine, option)

    batchs = image_batches(args.input, model.batch_size, True)

    logger.info(f"Infering data in {args.input}")
    for batch in track(batchs, description="[cyan]Processing batches", total=len(batchs)):
        images = [cv2.imread(image_path) for image_path in batch]

        results = model.predict(images)

        if args.output:
            for image_path, image, result in zip(batch, images, results):
                vis_image = visualize(image, result, args.labels)
                cv2.imwrite(str(output_dir / Path(image_path).name), vis_image)

    model.performance_report()
    logger.success("Finished Inference.")


if __name__ == '__main__':
    main()
