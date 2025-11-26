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
# File    :   classify.py
# Version :   6.4.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2025/01/23 14:38:16
# Desc    :   Classify 示例
# ==============================================================================
import argparse
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import supervision as sv

from trtyolo import TRTYOLO


class ClassificationAnnotator:
    """
    Ref: https://github.com/roboflow/supervision/commit/9c2aaa0d2ebac2fcb485f625a15059870deae1ee
    Annotate classification results on an image.
    """

    def __init__(
        self,
        color: Union[sv.Color, sv.ColorPalette] = sv.ColorPalette.DEFAULT,
        text_color: sv.Color = sv.Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: sv.Position = sv.Position.TOP_LEFT,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[sv.Color, sv.ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (sv.Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_position (sv.Position): Position of the text relative to the image.
                Possible values are defined in the `sv.Position` enum.
            color_lookup (sv.ColorLookup): Strategy for mapping colors to annotations.
                Options are `sv.ColorLookup.INDEX`, `sv.ColorLookup.CLASS`, `sv.ColorLookup.TRACE`.
        """
        self.color: Union[sv.Color, sv.ColorPalette] = color
        self.text_color: sv.Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_position: sv.Position = text_position
        self.color_lookup: sv.ColorLookup = color_lookup
        self.text_padding: int = text_padding

    def annotate(
        self,
        scene: np.ndarray,
        classifications: sv.Classifications,
        labels: List[str] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided classifications.
        Args:
            scene (np.ndarray): The image where labels will be drawn.
            classifications (Classifications): Object classifications to annotate.
            labels (List[str]): Optional. Custom labels for each classification.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.
        Returns:
            np.ndarray: The annotated image.
        Example:
            ```python
            >>> import supervision as sv
            >>> image = ...
            >>> classifications = sv.Classifications(...)
            >>> classification_annotator = sv.ClassificationAnnotator(text_position=sv.Position.CENTER)
            >>> annotated_frame = classification_annotator.annotate(
            ...     scene=image.copy(),
            ...     classifications=classifications
            ... )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        classification = classifications.get_top_k(k=1)
        classification_idx = classifications.class_id.tolist().index(classification[0][0])

        color = sv.annotators.utils.resolve_color(
            color=self.color,
            detections=classifications,
            detection_idx=classification_idx,
            color_lookup=self.color_lookup if custom_color_lookup is None else custom_color_lookup,
        )

        self.text_color = color

        text = f"{labels[classification_idx]} ({classifications.confidence[classification_idx] * 100:.2f}%)"

        text_wh = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=self.text_scale,
            thickness=self.text_thickness,
        )[0]

        if self.text_position == sv.Position.BOTTOM_LEFT:
            text_x = self.text_padding
            text_y = scene.shape[0] - self.text_padding
        elif self.text_position == sv.Position.BOTTOM_RIGHT:
            text_x = scene.shape[1] - text_wh[0] - self.text_padding
            text_y = scene.shape[0] - self.text_padding
        elif self.text_position == sv.Position.TOP_LEFT:
            text_x = self.text_padding
            text_y = text_wh[1] + self.text_padding
        elif self.text_position == sv.Position.TOP_RIGHT:
            text_x = scene.shape[1] - text_wh[0] - self.text_padding
            text_y = text_wh[1] + self.text_padding
        else:
            raise ValueError(f"Invalid position {self.text_position} for classification annotator.")

        cv2.putText(
            img=scene,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=self.text_scale,
            color=self.text_color.as_rgb(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )
        return scene


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
        classification_annotator = ClassificationAnnotator()
        class_name = [line.strip() for line in open(args.labels, "r")]

    input_path = Path(args.input)
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]

    model = TRTYOLO(args.engine, task="classify", swap_rb=True, profile=True)

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
                labels = [class_name[int(cls)] for cls in result.class_id]
                annotated_frame = classification_annotator.annotate(scene=image.copy(), classifications=result, labels=labels)
                output_file = output_dir / image_name
                cv2.imwrite(str(output_file), annotated_frame)
    elif input_path.is_file():
        file_ext = f"*.{input_path.suffix.lower()[1:]}"
        if file_ext not in extensions:
            raise ValueError(f"Unsupported file format: {input_path.suffix}")
        image = cv2.imread(str(input_path))
        result = model.predict(image)
        if args.output:
            labels = [class_name[int(cls)] for cls in result.class_id]
            annotated_frame = classification_annotator.annotate(scene=image.copy(), classifications=result, labels=labels)
            output_file = output_dir / input_path.name
            cv2.imwrite(str(output_file), annotated_frame)

    throughput, cpu_latency, gpu_latency = model.profile()
    print(throughput)
    print(cpu_latency)
    print(gpu_latency)


if __name__ == '__main__':
    main()
