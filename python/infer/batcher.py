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
# File    :   batcher.py
# Version :   2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/02/27 15:57:54
# Desc    :   Image Batcher.
# ==============================================================================
import random
from pathlib import Path
from typing import List, Union, Tuple, Iterator
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from python.utils import letterbox

__all__ = ['ImageBatcher']


class ImageBatcher:
    """
    Batches and preprocesses images for inference.

    Args:
        input_path (str): Path to the directory containing images or a single image file.
        batch_size (int): Size of each batch.
        imgsz (List[int]): Height and width dimensions of the input images.
        dtype (np.dtype): Data type of the input batch.
        dynamic (bool): Whether the batch size is dynamic.
        shuffle_files (bool, optional): Whether to shuffle the image files. Defaults to False.
    """
    def __init__(
        self, 
        input_path: str, 
        batch_size: int, 
        imgsz: List[int],
        dtype: np.dtype,
        dynamic: bool,
        shuffle_files: bool = False
    ) -> None:
        """
        Initialize the ImageBatcher instance.

        Args:
            input_path (str): Path to the directory containing images or a single image file.
            batch_size (int): Size of each batch.
            imgsz (List[int]): Height and width dimensions of the input images.
            dtype (np.dtype): Data type of the input batch.
            dynamic (bool): Whether the batch size is dynamic.
            shuffle_files (bool, optional): Whether to shuffle the image files. Defaults to False.
        """
        self.images = self._find_images(Path(input_path), shuffle_files)
        self.num_images = len(self.images)

        if self.num_images < 1:
            raise ValueError(f"No valid image files found in {input_path}")

        self.dtype = dtype
        self.dynamic = dynamic
        self.batch_size = batch_size
        self.height, self.width = imgsz

        # Subdivide the list of images into batches
        self.num_batches = 1 + int((self.num_images - 1) / self.batch_size)
        self.batches = [self.images[i * self.batch_size: (i + 1) * self.batch_size] for i in range(self.num_batches)]

    def __iter__(self) -> Iterator[Tuple[np.ndarray, List[Union[str, Path]], List[Tuple[int, int]]]]:
        """
        Iterator function to yield batches of preprocessed images.

        Yields:
            Tuple[np.ndarray, List[Union[str, Path]], List[Tuple[int, int]]]: Batch data, image paths, and image shape information.
        """
        for batch_images in self.batches:
            batch_shape = []
            batch_size = len(batch_images) if self.dynamic else self.batch_size
            batch_data = np.zeros((batch_size, 3, self.height, self.width), dtype=self.dtype)

            with ThreadPoolExecutor(max_workers=len(batch_images)) as executor:
                results = list(executor.map(self._preprocess_image, batch_images))

            for idx, (im, shape) in enumerate(results):
                batch_data[idx] = im
                batch_shape.append(shape)

            yield np.ascontiguousarray(batch_data), batch_images, batch_shape

    def _find_images(self, input_path: Path, shuffle_files: bool) -> None:
        """
        Find image files in the specified directory.

        Args:
            input_path (Path): Path to the directory containing images or a single image file.
            shuffle_files (bool): Whether to shuffle the image files.

        Returns:
            List[Path]: List of image file paths.
        """      
        if not input_path.exists():
            raise ValueError(f"Directory not found: {input_path}")

        extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        is_image = lambda file_path: file_path.is_file() and file_path.suffix.lower() in extensions

        images = sorted(file_path for file_path in input_path.iterdir() if is_image(file_path)) if input_path.is_dir() else [input_path]

        if shuffle_files:
            random.shuffle(images)

        if not images:
            raise ValueError(f"No image files found in {input_path}")

        return images

    def _preprocess_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocesses an image by reading, resizing, and normalizing.

        Args:
            image_path (str): The path to the input image file.

        Returns:
            Tuple[np.ndarray, Tuple[int, int]]: Preprocessed image, image shape.
        """
        # Read the image
        image = cv2.imread(str(image_path))

        # Resize and pad the image
        image, shape = letterbox(image, (self.height, self.width))
        
        # Convert color format and normalize pixel values
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(self.dtype) / 255.0
        
        # Transpose the image to CHW format
        image = np.transpose(image, (2, 0, 1))

        return image, shape