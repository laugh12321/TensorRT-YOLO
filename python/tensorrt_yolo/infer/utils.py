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
# File    :   utils.py
# Version :   6.0.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:06:46
# Desc    :   Utility functions for visualizing infernce results on images.
# ==============================================================================
import os
import sys
from collections.abc import Sequence, Set
from glob import glob

import cv2
import numpy as np
from loguru import logger

from .model import ClassifyRes, PoseRes, ResultType, RotatedBox, SegmentRes

__all__ = ["generate_labels", "visualize", "image_batches"]


def is_valid_image_path(path: str, image_extensions: Set[str]) -> bool:
    """Check if the given path is a valid image file."""
    return os.path.isfile(path) and path.lower().endswith(tuple(image_extensions))


def get_image_files_from_directory(data_path: str, image_extensions: Set[str]) -> Sequence[str]:
    """Get all image files from the specified directory."""
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(data_path, f'*{ext}'), recursive=True))
    return image_files


def validate_data_path(data_path: str, image_extensions: Set[str]) -> Sequence[str]:
    """Validate the data path and return the list of image files."""
    if os.path.isdir(data_path):
        image_files = get_image_files_from_directory(data_path, image_extensions)
        if not image_files:
            logger.error(f"No image files found in the directory '{data_path}'.")
            sys.exit(1)
    elif is_valid_image_path(data_path, image_extensions):
        image_files = [data_path]
    else:
        logger.error(f"The provided data path '{data_path}' is not a valid directory or a supported image file.")
        sys.exit(1)

    return image_files


def create_batches(image_files: Sequence[str], batch_size: int, pad_extra: bool) -> Sequence[Sequence[str]]:
    """Divide the image files into batches."""
    batches = [image_files[i : i + batch_size] for i in range(0, len(image_files), batch_size)]

    if pad_extra and batches and len(batches[-1]) < batch_size:
        last_image = batches[-1][-1]
        batches[-1].extend([last_image] * (batch_size - len(batches[-1])))

    return batches


def image_batches(
    data_path: str, batch_size: int, pad_extra: bool, image_extensions: Set[str] = {'.jpg', '.jpeg', '.png', '.bmp'}
) -> Sequence[Sequence[str]]:
    """
    Group image files in the specified directory into batches.

    Args:
        data_path (str): Path to the directory containing image files or a single image file.
        batch_size (int): Number of images per batch.
        pad_extra (bool): Whether to pad the last batch with extra images if they don't fit into a full batch.
        image_extensions (Set[str]): A set of file extensions to consider as image files.

    Returns:
        Sequence[Sequence[str]]: A list of batches, where each batch is a list of image file paths.

    Raises:
        ValueError: If the data path is not a valid directory or a supported image file.
        ValueError: If no image files are found in the directory.
    """
    image_files = validate_data_path(data_path, image_extensions)
    return create_batches(image_files, batch_size, pad_extra)


def generate_labels(labels_file: str) -> Sequence[str]:
    """
    Generate a list of labels based on the information in the labels file.

    Args:
        labels_file (str): Path to the labels file.

    Returns:
        Sequence[str]: List of label strings.
    """
    with open(labels_file) as f:
        return [label.strip() for label in f]


def xyxyr2xyxyxyxy(box: RotatedBox) -> Sequence[tuple[float, float]]:  # type: ignore
    """
    Convert a rotated bounding box to the coordinates of its four corners.

    Args:
        box (RotatedBox): The bounding box with left, top, right, bottom attributes,
        and a rotation angle (theta).

    Returns:
        Sequence[tuple[float, float]]: A list of four corner coordinates,
        each as an (x, y) tuple.
    """
    cos_value, sin_value = np.cos(box.theta), np.sin(box.theta)

    # Calculate the center coordinates of the box
    center_x, center_y = (box.left + box.right) * 0.5, (box.top + box.bottom) * 0.5

    # Calculate the half width and half height of the box
    half_width, half_height = (box.right - box.left) * 0.5, (box.bottom - box.top) * 0.5

    # Calculate corner vectors
    vec_x1, vec_y1 = half_width * cos_value, half_width * sin_value
    vec_x2, vec_y2 = half_height * sin_value, half_height * cos_value

    # Generate the four corners
    return [
        (center_x + vec_x1 - vec_x2, center_y + vec_y1 + vec_y2),
        (center_x + vec_x1 + vec_x2, center_y + vec_y1 - vec_y2),
        (center_x - vec_x1 + vec_x2, center_y - vec_y1 - vec_y2),
        (center_x - vec_x1 - vec_x2, center_y - vec_y1 + vec_y2),
    ]


def visualize(
    image: np.ndarray,
    result: ResultType,  # type: ignore
    labels: Sequence[str],  # List of label names
) -> np.ndarray:
    """
    Draw inference results on the input image and return the resulting image.

    Args:
        image (np.ndarray): The input image on which to draw inference results.
        result (ResultType): The inference result object.
        labels (Sequence[str]): A list of label names.

    Returns:
        np.ndarray: The image with drawn inference results.
    """
    # Image dimensions
    image_height, image_width, _ = image.shape

    # Line width, font size, and text thickness
    line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
    font_thickness = max(line_width - 1, 1)
    font_scale = line_width / 3

    # Colors
    lowlight_color = (253, 168, 208)
    mediumlight_color = (251, 81, 163)
    highlight_color = (125, 40, 81)

    # Copy the image to avoid modifying the original
    img = image.copy()

    # Pre-calculate skeleton connections
    skeleton_connections = [
        [16, 14],
        [14, 12],
        [17, 15],
        [15, 13],
        [12, 13],
        [6, 12],
        [7, 13],
        [6, 7],
        [6, 8],
        [7, 9],
        [8, 10],
        [9, 11],
        [2, 3],
        [1, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [5, 7],
    ]

    for i in range(result.num):
        label_text = f"{labels[result.classes[i]]} {result.scores[i]:.3f}"

        # Classification result
        if isinstance(result, ClassifyRes):
            text_position = [5, 32 + i * 32]
            cv2.putText(img, label_text, text_position, 0, font_scale, mediumlight_color, thickness=font_thickness, lineType=cv2.LINE_AA)
            continue

        # Bounding box (rotated or regular)
        if isinstance(result.boxes[i], RotatedBox):
            rotated_box_coords = xyxyr2xyxyxyxy(result.boxes[i])
            box_top_left = [int(coord) for coord in rotated_box_coords[0]]
            cv2.polylines(img, [np.asarray(rotated_box_coords, dtype=int)], True, mediumlight_color, line_width)
        else:
            box_coords = [
                int(coord) for coord in [result.boxes[i].left, result.boxes[i].top, result.boxes[i].right, result.boxes[i].bottom]
            ]
            box_top_left, box_bottom_right = (box_coords[0], box_coords[1]), (box_coords[2], box_coords[3])
            cv2.rectangle(img, box_top_left, box_bottom_right, mediumlight_color, thickness=line_width, lineType=cv2.LINE_AA)

        # Segmentation mask
        if isinstance(result, SegmentRes):
            resized_mask = cv2.resize(result.masks[i].to_numpy(), (image_width, image_height)) > 0
            box_mask = np.zeros_like(resized_mask, dtype=bool)
            box_mask[box_coords[1] : box_coords[3], box_coords[0] : box_coords[2]] = True
            resized_mask &= box_mask
            img[resized_mask] = img[resized_mask] * 0.5 + np.array(lowlight_color) * 0.5
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Pose keypoints
        if isinstance(result, PoseRes):
            confidence_threshold = 0.25
            for keypoint in result.kpts[i]:
                if keypoint.conf is None or keypoint.conf >= confidence_threshold:
                    cv2.circle(img, (int(keypoint.x), int(keypoint.y)), line_width, highlight_color, -1, lineType=cv2.LINE_AA)

            for connection in skeleton_connections:
                keypoint1 = result.kpts[i][connection[0] - 1]
                keypoint2 = result.kpts[i][connection[1] - 1]
                if keypoint1.conf >= confidence_threshold and keypoint2.conf >= confidence_threshold:
                    cv2.line(
                        img,
                        (int(keypoint1.x), int(keypoint1.y)),
                        (int(keypoint2.x), int(keypoint2.y)),
                        lowlight_color,
                        thickness=int(np.ceil(line_width / 2)),
                        lineType=cv2.LINE_AA,
                    )

        # Draw label background
        text_width, text_height = cv2.getTextSize(label_text, 0, fontScale=font_scale, thickness=font_thickness)[0]
        text_height += 3  # Padding

        is_text_outside = box_top_left[1] >= text_height
        if box_top_left[0] > image_width - text_width:
            box_top_left = image_width - text_width, box_top_left[1]

        box_bottom_right = box_top_left[0] + text_width, box_top_left[1] - text_height if is_text_outside else box_top_left[1] + text_height
        cv2.rectangle(img, box_top_left, box_bottom_right, highlight_color, -1, cv2.LINE_AA)

        text_position = (box_top_left[0], box_top_left[1] - 2 if is_text_outside else box_top_left[1] + text_height - 1)
        cv2.putText(img, label_text, text_position, 0, font_scale, lowlight_color, thickness=font_thickness, lineType=cv2.LINE_AA)

    return img
