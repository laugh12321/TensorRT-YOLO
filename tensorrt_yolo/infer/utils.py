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
# Version :   5.0.0
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

from .result import DetResult, OBBResult, PoseResult, RotatedBox, SegResult

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


def xyxyr2xyxyxyxy(box: RotatedBox) -> List[Tuple[float, float]]:  # type: ignore
    """
    Convert a rotated bounding box to the coordinates of its four corners.

    Args:
        box (RotatedBox): The bounding box with left, top, right, bottom attributes,
        and a rotation angle (theta).

    Returns:
        List[Tuple[float, float]]: A list of four corner coordinates,
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
        (center_x + vec_x1 - vec_x2, center_y + vec_y1 + vec_y2),
        (center_x + vec_x1 + vec_x2, center_y + vec_y1 - vec_y2),
        (center_x - vec_x1 + vec_x2, center_y - vec_y1 - vec_y2),
        (center_x - vec_x1 - vec_x2, center_y - vec_y1 + vec_y2),
    ]


def visualize(
    image: np.ndarray,
    result: Union[DetResult, OBBResult, SegResult, PoseResult],  # type: ignore
    labels: List[Tuple[str, Tuple[int, int, int]]],
) -> np.ndarray:
    """
    Draw inference results on the input image and return the resulting image.

    Args:
        image (np.ndarray): The input image on which to draw inference results.
        result (Union[DetResult, OBBResult, SegResult, PoseResult]): The inference result object.
        labels (List[Tuple[str, Tuple[int, int, int]]]): A list containing label names and their corresponding RGB color values.

    Returns:
        np.ndarray: The image with drawn inference results.
    """
    height, width, _ = image.shape
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)  # line width
    tf = max(lw - 1, 1)  # font thickness
    sf = lw / 3  # font scale

    # Pose
    if isinstance(result, PoseResult):
        conf_thres = 0.25
        radius = lw
        skeleton = [
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

    img = image.copy()
    for i in range(result.num):
        color = labels[result.classes[i]][1]

        # bounding box
        if isinstance(result.boxes[i], RotatedBox):
            box = xyxyr2xyxyxyxy(result.boxes[i])
            p1 = [int(b) for b in box[0]]
            cv2.polylines(img, [np.asarray(box, dtype=int)], True, color, lw)  # cv2 requires nparray box
        else:
            box = list(map(int, [result.boxes[i].left, result.boxes[i].top, result.boxes[i].right, result.boxes[i].bottom]))
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

        # mask
        if isinstance(result, SegResult):
            # Resize the segmentation mask to match the image dimensions and convert to a boolean mask
            mask = cv2.resize(result.masks[i], (width, height)) > 0

            # Create a boolean mask for the bounding box area
            box_mask = np.zeros_like(mask, dtype=bool)
            box_mask[box[1] : box[3], box[0] : box[2]] = True

            # Combine the segmentation mask with the bounding box mask
            mask &= box_mask

            # Blend the mask color with the image only within the masked area
            img[mask] = img[mask] * 0.5 + np.array(color) * 0.5

            # Clip the values to valid range and ensure the result is an unsigned 8-bit integer
            img = np.clip(img, 0, 255).astype(np.uint8)

        # keypoint
        if isinstance(result, PoseResult):
            nkpt = len(result.kpts[i])
            is_pose = nkpt == 17
            kpt_line = is_pose  # `kpt_line=True` for now only supports human pose plotting
            for kpt in result.kpts[i]:
                if kpt.x % width != 0 and kpt.y % height != 0:
                    if kpt.conf is not None and kpt.conf < conf_thres:
                        continue
                    cv2.circle(img, (int(kpt.x), int(kpt.y)), radius, color, -1, lineType=cv2.LINE_AA)

            if kpt_line:
                for sk in skeleton:
                    kpt1 = result.kpts[i][sk[0] - 1]
                    kpt2 = result.kpts[i][sk[1] - 1]

                    if kpt1.conf < conf_thres or kpt2.conf < conf_thres:
                        continue
                    if kpt1.x % width == 0 or kpt1.y % height == 0 or kpt1.x < 0 or kpt1.y < 0:
                        continue
                    if kpt2.x % width == 0 or kpt2.y % height == 0 or kpt2.x < 0 or kpt2.y < 0:
                        continue
                    cv2.line(
                        img,
                        (int(kpt1.x), int(kpt1.y)),
                        (int(kpt2.x), int(kpt2.y)),
                        color,
                        thickness=int(np.ceil(lw / 2)),
                        lineType=cv2.LINE_AA,
                    )

        # label
        label = f"{labels[result.classes[i]][0]} {result.scores[i]:.2f}"
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        h += 3  # add pixels to pad text
        outside = p1[1] >= h  # label fits outside box
        if p1[0] > width - w:  # shape is (h, w), check if label extend beyond right side of image
            p1 = width - w, p1[1]
        p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),
            0,
            sf,
            (255, 255, 255),
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

    return img
