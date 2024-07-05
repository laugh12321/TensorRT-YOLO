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
# File    :   detection.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/03 14:06:55
# Desc    :   This script provides classes for deploying detection models.
# ==============================================================================
from typing import Any, List, Union

from .. import c_lib_wrap as C

__all__ = ["DeployDet", "DeployCGDet", "DetectionResult", "Box"]

DetectionResult = C.result.DetectionResult
Box = C.result.Box


class DeployDet:
    def __init__(self, engine_file: str, use_cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployDet class with the given engine file, optional CUDA memory usage, and device index.

        Args:
            engine_file (str): Path to the engine file.
            use_cuda_memory (bool, optional): Whether to use CUDA memory. Defaults to False.
            device (int, optional): Device index to use. Defaults to 0.
        """
        self._model = C.detection.DeployDet(engine_file, use_cuda_memory, device)

    @property
    def batch(self) -> int:
        """
        Get the batch size.

        Returns:
            int: Batch size.
        """
        return self._model.batch

    def predict(self, images: Union[Any, List[Any]]) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Predict the detection results for the given images.

        Args:
            images (Union[Any, List[Any]]): A single image or a list of images.

        Returns:
            Union[DetectionResult, List[DetectionResult]]: Detection result or a list of detection results.
        """
        return self._model.predict(images)


class DeployCGDet:
    def __init__(self, engine_file: str, use_cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployCGDet class with the given engine file, optional CUDA memory usage, and device index.

        Args:
            engine_file (str): Path to the engine file.
            use_cuda_memory (bool, optional): Whether to use CUDA memory. Defaults to False.
            device (int, optional): Device index to use. Defaults to 0.
        """
        self._model = C.detection.DeployCGDet(engine_file, use_cuda_memory, device)

    @property
    def batch(self) -> int:
        """
        Get the batch size.

        Returns:
            int: Batch size.
        """
        return self._model.batch

    def predict(self, images: Union[Any, List[Any]]) -> Union[DetectionResult, List[DetectionResult]]:
        """
        Predict the detection results for the given images.

        Args:
            images (Union[Any, List[Any]]): A single image or a list of images.

        Returns:
            Union[DetectionResult, List[DetectionResult]]: Detection result or a list of detection results.
        """
        return self._model.predict(images)
