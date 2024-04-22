#!/usr/bin/env python
# ==============================================================================
# Copyright (c) 2024 laugh12321 Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# File    :   general.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/21 14:10:04
# Desc    :   General utils.
# ==============================================================================
import random
from typing import List, Tuple, Union

import cv2
import numpy as np

from .structs import DetectInfo

__all__ = ['generate_labels_with_colors', 'letterbox', 'scale_boxes', 'visualize_detections']


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


def letterbox(
    image: np.ndarray, new_shape: Union[Tuple[int, int], int], color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resizes and pads the input image to the specified new shape.

    Args:
        image (np.ndarray): The input image.
        new_shape (Union[Tuple[int, int], int]): The target shape for resizing.
        color (Tuple[int, int, int], optional): The color used for padding. Defaults to (114, 114, 114).

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: Resized image, image origal shape.
    """
    shape = image.shape[:2]  # Current shape [height, width]

    # If new_shape is an integer, convert it to a tuple
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Compute the scaling ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

    # Resize the image
    if shape != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    # Add border to the image for padding
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, shape


def scale_boxes(boxes: np.ndarray, input_shape: Tuple[int, int], output_shape: Tuple[int, int]) -> np.ndarray:
    """
    Rescales (xyxy) bounding boxes from input_shape to output_shape.

    Args:
        boxes (np.ndarray): Input bounding boxes in (xyxy) format.
        input_shape (Tuple[int, int]): Source shape (height, width).
        output_shape (Tuple[int, int]): Target shape (height, width).

    Returns:
        np.ndarray: Rescaled bounding boxes.
    """
    gain = min(input_shape[0] / output_shape[0], input_shape[1] / output_shape[1])  # gain  = old / new
    pad = (input_shape[1] - output_shape[1] * gain) / 2, (input_shape[0] - output_shape[0] * gain) / 2  # wh padding

    # Adjust for padding
    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding

    # Rescale using the ratio
    boxes[..., :4] /= gain

    # Clip coordinates to be within the target shape
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, output_shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, output_shape[0])  # y1, y2

    return boxes


def visualize_detections(image_path: str, output_path: str, det_info: DetectInfo, labels: List[Tuple[str, Tuple]]) -> None:
    """
    Visualize object detections on the input image and save the result.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image.
        det_info (DetectInfo): Object containing detection information (num, boxes, classes, scores).
        labels (List[Tuple[str, Tuple]]): List of label names and corresponding colors.
    """
    image = cv2.imread(image_path)

    for box, class_, score in zip(det_info.boxes, det_info.classes, det_info.scores):
        box = list(map(int, box))
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

    cv2.imwrite(output_path, image)
