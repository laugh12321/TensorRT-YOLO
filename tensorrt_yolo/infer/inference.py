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
# File    :   inference.py
# Version :   5.0.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/03 14:06:55
# Desc    :   Classes for deploying YOLO models, including
#             detection, OBB, segmentation and pose estimation.
# ==============================================================================
from typing import Any, List, Union

from .. import c_lib_wrap as C
from .result import DetResult, OBBResult, PoseResult, SegResult

__all__ = ["DeployDet", "DeployCGDet", "DeployOBB", "DeployCGOBB", "DeploySeg", "DeployCGSeg", "DeployPose", "DeployCGPose"]


class BaseDeploy:
    def __init__(self, engine_file: str, model_class: Any, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Base class for model deployment with common functionality.

        Args:
            engine_file (str): Path to the engine file.
            model_class (Any): The model class for the specific type (e.g., DeployDet, DeployCGDet, etc.).
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        self._model = model_class(engine_file, cuda_memory, device)

    @property
    def batch(self) -> int:
        """
        Get the batch size supported by the model.

        Returns:
            int: Batch size.
        """
        return self._model.batch

    def predict(
        self, images: Union[Any, List[Any]]
    ) -> Union[DetResult, List[DetResult], OBBResult, List[OBBResult], SegResult, List[SegResult], PoseResult, List[PoseResult]]:  # type: ignore
        """
        Predict the detection results for the given images.

        Args:
            images (Union[Any, List[Any]]): A single image or a list of images for which to predict object detections.

        Returns:
            Union[DetResult, List[DetResult], OBBResult, List[OBBResult], SegResult, List[SegResult], PoseResult, List[PoseResult]]:
            The detection result for the image or a list of detection results for multiple images.
        """
        return self._model.predict(images)


class DeployDet(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployDet class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployDet, cuda_memory, device)


class DeployCGDet(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployCGDet class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployCGDet, cuda_memory, device)


class DeployOBB(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployOBB class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployOBB, cuda_memory, device)


class DeployCGOBB(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployCGOBB class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployCGOBB, cuda_memory, device)


class DeploySeg(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeploySeg class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeploySeg, cuda_memory, device)


class DeployCGSeg(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployCGSeg class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployCGSeg, cuda_memory, device)


class DeployPose(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeploySeg class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployPose, cuda_memory, device)


class DeployCGPose(BaseDeploy):
    def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Initialize the DeployCGSeg class with the given engine file.

        Args:
            engine_file (str): Path to the engine file.
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        super().__init__(engine_file, C.inference.DeployCGPose, cuda_memory, device)
