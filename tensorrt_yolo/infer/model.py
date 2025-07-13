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
# File    :   model.py
# Version :   6.0.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2025/01/19 17:50:16
# Desc    :   Define base model class and factory function for model creation and prediction
# ==============================================================================
from typing import Sequence, Type, TypeAlias, Union

from cv2.typing import MatLike
from loguru import logger

from .. import c_lib_wrap as C

__all__ = [
    "Mask",
    "KeyPoint",
    "Box",
    "RotatedBox",
    "ClassifyRes",
    "DetectRes",
    "OBBRes",
    "SegmentRes",
    "PoseRes",
    "InferOption",
    "ResultType",
    "ClassifyModel",
    "DetectModel",
    "OBBModel",
    "SegmentModel",
    "PoseModel",
]

# Define type aliases
Mask: TypeAlias = C.result.Mask
KeyPoint: TypeAlias = C.result.KeyPoint
Box: TypeAlias = C.result.Box
RotatedBox: TypeAlias = C.result.RotatedBox
ClassifyRes: TypeAlias = C.result.ClassifyRes
DetectRes: TypeAlias = C.result.DetectRes
OBBRes: TypeAlias = C.result.OBBRes
SegmentRes: TypeAlias = C.result.SegmentRes
PoseRes: TypeAlias = C.result.PoseRes
InferOption: TypeAlias = C.option.InferOption

# Define union type
ResultType: TypeAlias = Union[ClassifyRes, DetectRes, OBBRes, SegmentRes, PoseRes]


class BaseModel:
    """
    Base model class for different deployment classes.
    """

    def __init__(self, deploy_class: Type, engine_file: str, option: InferOption) -> None:
        """
        Initialize the base model.

        Args:
            deploy_class (Type): Deployment class.
            engine_file (str): Engine file.
            option (InferOption): Inference option.
        """
        self._model = deploy_class(engine_file, option)

    @property
    def batch_size(self) -> int:
        """
        Get the maximum batch size of the model.

        Returns:
            int: Batch size.
        """
        return self._model.batch_size()

    def performance_report(self) -> None:
        """
        Get the performance report of the model.

        Returns:
            tuple[str, str, str]: Performance report.
        """
        throughput, cpu_latency, gpu_latency = self._model.performance_report()
        if throughput != "":
            logger.success("=== Performance summary ===")
            logger.success(throughput)
            logger.success(cpu_latency)
            logger.success(gpu_latency)

    def predict(self, images: Union[MatLike, Sequence[MatLike]]) -> Union[ResultType, Sequence[ResultType]]:
        """
        Predict the inference results for the given images.

        Args:
            images (Union[MatLike, Sequence[MatLike]]): Single image or list of images.

        Returns:
            Union[ResultType, Sequence[ResultType]]: Inference result or list of inference results.
        """
        return self._model.predict(images)

    def clone(self) -> 'BaseModel':
        """
        Clone the current model.

        Returns:
            BaseModel: A new instance of BaseModel that is a clone of the current model.
        """

        class CloneModel(BaseModel):
            def __init__(self, model):
                self._model = model

        return CloneModel(self._model.clone())


def create_model_class(deploy_class: Type) -> Type[BaseModel]:
    """
    Factory function to create a specific deployment class.

    Args:
        deploy_class (Type): Specific deployment class.

    Returns:
        Type[BaseModel]: New deployment class.
    """

    class Model(BaseModel):
        def __init__(self, engine_file: str, option: InferOption) -> None:
            """
            Initialize the model.

            Args:
                engine_file (str): Engine file.
                option (InferOption): Inference option.
            """
            super().__init__(deploy_class, engine_file, option)

    return Model


ClassifyModel = create_model_class(C.model.ClassifyModel)
DetectModel = create_model_class(C.model.DetectModel)
OBBModel = create_model_class(C.model.OBBModel)
SegmentModel = create_model_class(C.model.SegmentModel)
PoseModel = create_model_class(C.model.PoseModel)
