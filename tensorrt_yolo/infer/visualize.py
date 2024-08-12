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
# Version :   2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:06:46
# Desc    :   Utility functions for visualizing object detection results on images.
# ==============================================================================
import random
from typing import List, Tuple

import cv2
import numpy as np

from .result import Box, DetectionResult

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


def xyxyr2xyxyxyxy(box: Box) -> List[Tuple[int, int]]:  # type: ignore
    """
    Convert a rotated bounding box to the coordinates of its four corners.

    Args:
        box (Box): The bounding box with left, top, right, bottom attributes,
        and a rotation angle (theta).

    Returns:
        List[Tuple[int, int]]: A list of four corner coordinates,
        each as an (x, y) tuple.
    """
    # Calculate the cosine and sine of the angle
    cos_value = np.cos(box.theta)
    sin_value = np.sin(box.theta)

    # Calculate the center coordinates of the box
    center_x = (box.left + box.right) * 0.5
    center_y = (box.top + box.bottom) * 0.5

    # Calculate the half width and half height of the box
    half_width = (box.right - box.left) * 0.5
    half_height = (box.bottom - box.top) * 0.5

    # Calculate the rotated corner vectors
    vec_x1 = half_width * cos_value
    vec_y1 = half_width * sin_value
    vec_x2 = half_height * sin_value
    vec_y2 = half_height * cos_value

    # Return the four corners of the rotated rectangle
    return [
        (int(center_x + vec_x1 - vec_x2), int(center_y + vec_y1 + vec_y2)),
        (int(center_x + vec_x1 + vec_x2), int(center_y + vec_y1 - vec_y2)),
        (int(center_x - vec_x1 + vec_x2), int(center_y - vec_y1 - vec_y2)),
        (int(center_x - vec_x1 - vec_x2), int(center_y - vec_y1 + vec_y2)),
    ]


def visualize_detections(
    image: np.ndarray,
    det_result: DetectionResult,  # type: ignore
    labels: List[Tuple[str, Tuple[int, int, int]]],
    is_obb: bool,
) -> np.ndarray:
    """
    Visualize object detections on the input image and return the result.

    Args:
        image (np.ndarray): The input image on which to visualize detections.
        det_result (DetectionResult): Object containing detection results, including the number of detections,
                                      bounding boxes, class indices, and scores.
        labels (List[Tuple[str, Tuple[int, int, int]]]): A list of label names and their corresponding RGB colors.
                                                         Each element is a tuple where the first item is the label name
                                                         and the second is the color tuple (R, G, B).
        is_obb (bool): A flag indicating whether the bounding boxes are oriented (rotated) bounding boxes (OBB).
                       If True, the bounding boxes are treated as rotated rectangles.

    Returns:
        np.ndarray: The image with visualized detections, including labeled bounding boxes or rotated rectangles.
    """
    for box, class_, score in zip(det_result.boxes, det_result.classes, det_result.scores):
        color = labels[class_][1]
        label_text = f"{labels[class_][0]} {score:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        if is_obb:
            corners = xyxyr2xyxyxyxy(box)
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))

            # Draw label text
            label_top_left = (corners[0][0][0], corners[0][0][1] - label_size[1])
            label_bottom_right = (corners[0][0][0] + label_size[0], corners[0][0][1])
            cv2.rectangle(image, label_top_left, label_bottom_right, color, thickness=-1)
            cv2.putText(image, label_text, (corners[0][0][0], corners[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Draw polylines for rotated bounding box
            corners = np.array(corners, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [corners], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
        else:
            box = list(map(int, [box.left, box.top, box.right, box.bottom]))

            # Draw label text
            label_rect = (box[0], box[1], label_size[0], label_size[1])
            label_rect = tuple(map(int, label_rect))
            cv2.rectangle(image, label_rect[:2], (label_rect[0] + label_rect[2], label_rect[1] + label_rect[3]), color, -1)
            cv2.putText(
                image, label_text, (label_rect[0], label_rect[1] + label_size[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )

            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, thickness=1, lineType=cv2.LINE_AA)

    return image
