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
# Version :   5.1.1
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/03 14:06:55
# Desc    :   Classes for deploying YOLO models, including
#             detection, OBB, segmentation and pose estimation.
# ==============================================================================
from collections.abc import Sequence
from typing import Union

from cv2.typing import MatLike

from .. import c_lib_wrap as C
from .result import ResultType

__all__ = [
    'DeployCGCls',
    'DeployCGDet',
    'DeployCGOBB',
    'DeployCGPose',
    'DeployCGSeg',
    'DeployCls',
    'DeployDet',
    'DeployOBB',
    'DeployPose',
    'DeploySeg',
]


class BaseDeploy:
    def __init__(self, engine_file: str, deploy_class: type, cuda_memory: bool = False, device: int = 0) -> None:
        """
        Base class for model deployment with common functionality.

        Args:
            engine_file (str): Path to the engine file.
            deploy_class (type): The deploy class for the specific type (e.g., DeployDet, DeployCGDet, etc.).
            cuda_memory (bool, optional): Flag indicating whether input data resides in GPU memory (true) or CPU memory (false). Defaults to false.
            device (int, optional): Device index for the inference. Defaults to 0.
        """
        self._model = deploy_class(engine_file, cuda_memory, device)

    @property
    def batch(self) -> int:
        """
        Get the batch size supported by the model.

        Returns:
            int: Batch size.
        """
        return self._model.batch

    def predict(self, images: Union[MatLike, Sequence[MatLike]]) -> Union[ResultType, Sequence[ResultType]]:  # type: ignore
        """
        Predict the inference results for the given images.

        Args:
            images (Union[MatLike, Sequence[MatLike]]): A single image or a list of images for which to predict inference.

        Returns:
            Union[ResultType, Sequence[ResultType]]: The inference result for the image or a list of inference results for multiple images.
        """
        return self._model.predict(images)


def create_deploy_class(deploy_class: type) -> type[BaseDeploy]:
    """
    Factory function to create a specific deployment class.

    Args:
        deploy_class (type): The deploy class for the specific type.

    Returns:
        type[BaseDeploy]: A new deployment class.
    """

    class Deploy(BaseDeploy):
        def __init__(self, engine_file: str, cuda_memory: bool = False, device: int = 0) -> None:
            super().__init__(engine_file, deploy_class, cuda_memory, device)

    return Deploy


DeployCls = create_deploy_class(C.inference.DeployCls)
DeployCGCls = create_deploy_class(C.inference.DeployCGCls)
DeployDet = create_deploy_class(C.inference.DeployDet)
DeployCGDet = create_deploy_class(C.inference.DeployCGDet)
DeployOBB = create_deploy_class(C.inference.DeployOBB)
DeployCGOBB = create_deploy_class(C.inference.DeployCGOBB)
DeployPose = create_deploy_class(C.inference.DeployPose)
DeployCGPose = create_deploy_class(C.inference.DeployCGPose)
DeploySeg = create_deploy_class(C.inference.DeploySeg)
DeployCGSeg = create_deploy_class(C.inference.DeployCGSeg)
