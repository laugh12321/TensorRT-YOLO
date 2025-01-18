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
# File    :   result.py
# Version :   5.1.1
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/08/04 12:39:02
# Desc    :   Wrapper for result classes from the C library.
# ==============================================================================
from typing import Union

from .. import c_lib_wrap as C

__all__ = ['Box', 'RotatedBox', 'KeyPoint', 'ClsResult', 'DetResult', 'OBBResult', 'PoseResult', 'SegResult', 'ResultType']


Box: type[C.result.Box] = C.result.Box
RotatedBox: type[C.result.RotatedBox] = C.result.RotatedBox
KeyPoint: type[C.result.KeyPoint] = C.result.KeyPoint
ClsResult: type[C.result.ClsResult] = C.result.ClsResult
DetResult: type[C.result.DetResult] = C.result.DetResult
OBBResult: type[C.result.OBBResult] = C.result.OBBResult
PoseResult: type[C.result.PoseResult] = C.result.PoseResult
SegResult: type[C.result.SegResult] = C.result.SegResult

ResultType = Union[ClsResult, DetResult, OBBResult, PoseResult, SegResult]
