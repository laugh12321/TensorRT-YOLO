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
# Version :   3.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:06:46
# Desc    :   Utility functions for visualizing infernce results on images.
# ==============================================================================
import os
import random
import sys
from glob import glob
from typing import List, Set, Tuple, Union

import cv2
import numpy as np
from loguru import logger

from .result import DetResult, OBBResult, RotatedBox

__all__ = ["generate_labels_with_colors", "visualize", "image_batches"]


def image_batches(
    data_path: str, batch_size: int, pad_extra: bool, image_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp'}
) -> List[List[str]]:
    """
    Group image files in the specified directory into batches.

    Args:
        data_path (str): Path to the directory containing image files or a single image file.
        batch_size (int): Number of images per batch.
        pad_extra (bool): Whether to pad the last batch with extra images if they don't fit into a full batch.
        image_extensions (Set[str]): A set of file extensions to consider as image files.

    Returns:
        List[List[str]]: A list of batches, where each batch is a list of image file paths.

    Raises:
        ValueError: If the data path is not a valid directory or a supported image file.
        ValueError: If no image files are found in the directory.
    """
    # Check if the data path is a valid directory or a supported image file
    if not (os.path.isdir(data_path) or (os.path.isfile(data_path) and data_path.lower().endswith(tuple(image_extensions)))):
        logger.error(f"The provided data path '{data_path}' is not a valid directory or a supported image file.")
        sys.exit(1)

    # Initialize the list to store image file paths
    image_files = []

    # If it's a directory, find all image files with the specified extensions
    if os.path.isdir(data_path):
        # Use glob to find all image files with the specified extensions
        image_files = [file for ext in image_extensions for file in glob(os.path.join(data_path, f'*{ext}'), recursive=True)]
        # Raise an error if no image files are found
        if not image_files:
            logger.error(f"No image files found in the directory '{data_path}'.")
            sys.exit(1)
    else:
        # If it's a single image file, add it to the list
        image_files.append(data_path)

    # Divide the image files into batches
    batches = [image_files[i : i + batch_size] for i in range(0, len(image_files), batch_size)]

    # Handle extra images if necessary
    if pad_extra and batches and len(batches[-1]) < batch_size:
        last_image = batches[-1][-1]
        batches[-1].extend([last_image] * (batch_size - len(batches[-1])))

    return batches


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


def xyxyr2xyxyxyxy(box: RotatedBox) -> List[Tuple[int, int]]:  # type: ignore
    """
    Convert a rotated bounding box to the coordinates of its four corners.

    Args:
        box (RotatedBox): The bounding box with left, top, right, bottom attributes,
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


def visualize(
    image: np.ndarray,
    result: Union[DetResult, OBBResult],  # type: ignore
    labels: List[Tuple[str, Tuple[int, int, int]]],
) -> np.ndarray:
    """
    Draw inference results on the input image and return the resulting image.

    Args:
        image (np.ndarray): The input image on which to draw inference results.
        result (Union[DetResult, OBBResult]): The inference result object.
        labels (List[Tuple[str, Tuple[int, int, int]]]): A list containing label names and their corresponding RGB color values.

    Returns:
        np.ndarray: The image with drawn inference results.
    """
    for i in range(result.num):
        color = labels[result.classes[i]][1]
        label_text = f"{labels[result.classes[i]][0]} {result.scores[i]:.2f}"
        label_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        if isinstance(result, OBBResult):
            corners = xyxyr2xyxyxyxy(result.boxes[i])
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
            box = list(map(int, [result.boxes[i].left, result.boxes[i].top, result.boxes[i].right, result.boxes[i].bottom]))

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
