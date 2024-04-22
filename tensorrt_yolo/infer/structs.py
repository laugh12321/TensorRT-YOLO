#!/usr/bin/env python
# -*-coding:utf-8 -*-
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
# File    :   structs.py
# Version :   3.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/21 14:08:33
# Desc    :   Defines data classes.
# ==============================================================================
from typing import List, Tuple, Union
from dataclasses import dataclass

import numpy as np

from .common import HostDeviceMem

__all__ = ['TensorInfo', 'DetectInfo']


@dataclass
class TensorInfo:
    """
    A data class representing information about a tensor.

    Attributes:
        name (str): The name of the tensor.
        shape (Tuple[int]): The shape of the tensor.
        input (bool): Indicates whether the tensor is an input tensor or not.
        memory (HostDeviceMem): The memory information of the tensor, containing both host and device memory.

    Note:
        HostDeviceMem is a placeholder for the type of memory allocation object.
    """
    name: str
    input: bool
    shape: Tuple[int]
    memory: HostDeviceMem


@dataclass
class DetectInfo:
    """
    A data class representing information about detections.

    Attributes:
        num (int): The number of detections.
        boxes (Union[List, np.ndarray]): The bounding boxes of the detections.
        scores (Union[List, np.ndarray]): The confidence scores of the detections.
        classes (Union[List, np.ndarray]): The class indices of the detections.
    """
    num: int
    boxes: Union[List, np.ndarray]
    scores: Union[List, np.ndarray]
    classes: Union[List, np.ndarray]
