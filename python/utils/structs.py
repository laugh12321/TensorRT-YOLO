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
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/21 14:08:33
# Desc    :   Defines data classes.
# ==============================================================================
from typing import List, Union
import numpy as np
from dataclasses import dataclass
from pycuda.gpuarray import GPUArray

__all__ = ['TensorInfo', 'DetectInfo']


@dataclass
class TensorInfo:
    name: str
    shape: tuple
    dtype: np.dtype
    host: np.ndarray
    device: GPUArray


@dataclass
class DetectInfo:
    num: int
    boxes: Union[List, np.ndarray]
    scores: Union[List, np.ndarray]
    classes: Union[List, np.ndarray]
