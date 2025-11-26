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
# Version :   6.4.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2025/01/23 14:16:11
# Desc    :   Detect 示例
# ==============================================================================
import argparse
from pathlib import Path

import cv2
import supervision as sv

from trtyolo import TRTYOLO


def main():
    parser = argparse.ArgumentParser(description='YOLO Series Inference For Segmentation.')
    parser.add_argument('-e', '--engine', required=True, type=str, help='The serialized TensorRT engine.')
    parser.add_argument('-i', '--input', required=True, type=str, help="Path to the image or directory to process.")
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory where to save the visualization results.')
    parser.add_argument(
        "-l", "--labels", default="./labels.txt", help="File to use for reading the class labels from, default: ./labels.txt"
    )

    args = parser.parse_args()

    if args.output and not args.labels:
        raise ValueError("Please provide a labels file using -l or --labels.")

    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        class_name = [line.strip() for line in open(args.labels, "r")]

    input_path = Path(args.input)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]

    model = TRTYOLO(args.engine, task="detect", profile=True, swap_rb=True)

    if input_path.is_dir():
        images, image_names = [], []
        for ext in extensions:
            images.extend([cv2.imread(str(image_path)) for image_path in input_path.glob(ext)])
            image_names.extend([image_path.name for image_path in input_path.glob(ext)])
        if not images:
            raise ValueError(f"No images found in directory: {input_path}")
        results = model.predict(images)
        if args.output:
            for image_name, image, result in zip(image_names, images, results):
                labels = [f"{class_name[int(cls)]} {conf:.2f}" for cls, conf in zip(result.class_id, result.confidence)]
                annotated_frame = box_annotator.annotate(scene=image.copy(), detections=result)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=result, labels=labels)
                output_file = output_dir / image_name
                cv2.imwrite(str(output_file), annotated_frame)
    elif input_path.is_file():
        file_ext = f"*.{input_path.suffix.lower()[1:]}"
        if file_ext not in extensions:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        image = cv2.imread(str(input_path))
        result = model.predict(image)
        if args.output:
            labels = [f"{class_name[int(cls)]} {conf:.2f}" for cls, conf in zip(result.class_id, result.confidence)]
            annotated_frame = box_annotator.annotate(scene=image.copy(), detections=result)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=result, labels=labels)
            output_file = output_dir / input_path.name
            cv2.imwrite(str(output_file), annotated_frame)

    throughput, cpu_latency, gpu_latency = model.profile()
    print(throughput)
    print(cpu_latency)
    print(gpu_latency)


if __name__ == '__main__':
    main()
