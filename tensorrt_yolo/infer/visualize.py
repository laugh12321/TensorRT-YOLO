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
# File    :   visualize.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:06:46
# Desc    :   Utility functions for visualizing object detection results on images.
# ==============================================================================
import random
from typing import List, Tuple

import cv2
import numpy as np

from .detection import DetectionResult

__all__ = ["generate_labels_with_colors", "visualize_detections"]


def generate_labels_with_colors(labels_file: str) -> List[Tuple[str, Tuple[int, int, int]]]:
    """
    Generate labels with random RGB colors based on the information in the labels file.

    Args:
        labels_file (str): Path to the labels file.

    Returns:
        List[Tuple[str, Tuple[int, int, int]]]: List of label-color tuples.
    """

    def generate_random_rgb() -> Tuple[int, int, int]:
        """
        Generate a random RGB color tuple.

        Returns:
            Tuple[int, int, int]: Random RGB color tuple.
        """
        return tuple(random.randint(0, 255) for _ in range(3))

    with open(labels_file) as f:
        return [(label.strip(), generate_random_rgb()) for label in f]


def visualize_detections(image: np.ndarray, det_result: DetectionResult, labels: List[Tuple[str, Tuple[int, int, int]]]) -> np.ndarray:
    """
    Visualize object detections on the input image and return the result.

    Args:
        image (np.ndarray): The input image on which to visualize detections.
        det_result (DetectionResult): Object containing detection result (num, boxes, classes, scores).
        labels (List[Tuple[str, Tuple[int, int, int]]]): List of label names and corresponding colors.

    Returns:
        np.ndarray: The image with visualized detections.
    """
    for box, class_, score in zip(det_result.boxes, det_result.classes, det_result.scores):
        box = list(map(int, [box.left, box.top, box.right, box.bottom]))
        color = labels[class_][1]
        label_text = f"{labels[class_][0]} {score:.2f}"

        # Draw rectangle with corners
        pad = min(int((box[2] - box[0]) / 6), int((box[3] - box[1]) / 6))
        points = [
            ((box[0], box[1]), (box[0] + pad, box[1])),
            ((box[0], box[1]), (box[0], box[1] + pad)),
            ((box[2], box[1]), (box[2] - pad, box[1])),
            ((box[2], box[1]), (box[2], box[1] + pad)),
            ((box[0], box[3]), (box[0] + pad, box[3])),
            ((box[0], box[3]), (box[0], box[3] - pad)),
            ((box[2], box[3]), (box[2] - pad, box[3])),
            ((box[2], box[3]), (box[2], box[3] - pad)),
        ]
        for corner in points:
            cv2.line(image, *corner, color, 2, cv2.LINE_AA)

        # Draw label text
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        label_rect = (box[0], box[1], label_size[0], label_size[1])
        label_rect = tuple(map(int, label_rect))
        cv2.rectangle(image, label_rect[:2], (label_rect[0] + label_rect[2], label_rect[1] + label_rect[3]), color, -1)
        cv2.putText(image, label_text, (label_rect[0], label_rect[1] + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Draw rectangle
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness=1, lineType=cv2.LINE_AA)

        # Draw rectangle mask
        mask = image.copy()
        cv2.rectangle(mask, (box[0], box[1]), (box[2], box[3]), color, thickness=-1)
        image = cv2.addWeighted(image, 0.8, mask, 0.2, 0)

    return image
