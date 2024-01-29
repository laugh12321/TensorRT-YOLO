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
# File    :   utils.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/21 14:10:04
# Desc    :   Utility functions for object detection timers and visualization.
# ==============================================================================
import cv2
import time
import numpy as np
from typing import Tuple
from .structs import DetectInfo

__all__ = ['Timer', 'visualize']


class Times:
    def __init__(self):
        self._time = 0
        self._st = 0

    @property
    def time(self):
        return round(self._time, 4)

    def start(self):
        self._st = time.perf_counter_ns()

    def end(self):
        self._time += (time.perf_counter_ns() - self._st) / 1e6  # Convert to milliseconds

    def reset(self):
        self._time = 0
        self._st = 0


class Timer:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.infer_count = 0
        self.pre_time = Times()
        self.infer_time = Times()
        self.post_time = Times()

    def info(self):
        total_time = sum([t.time for t in [self.pre_time, self.infer_time, self.post_time]])
        average_latency = self._avg_time(total_time)

        print("------------------ Inference Time Info ----------------------")
        print(f"total_time (ms): {total_time:.2f}, batch_infer_count: {self.infer_count}, batch_size: {self.batch_size}")
        print(f"total latency time (ms): {average_latency:.2f}", end=", ")
        print(f"preprocess latency time (ms): {self._avg_time(self.pre_time.time):.2f}", end=", ")
        print(f"inference latency time (ms): {self._avg_time(self.infer_time.time):.2f}", end=", ")
        print(f"postprocess latency time (ms): {self._avg_time(self.post_time.time):.2f}")

    def _avg_time(self, time):
        return round(time / max(1, self.infer_count), 4)


def visualize(image: np.ndarray, det_info: DetectInfo, color: Tuple[int, int, int] = (210, 127, 146)):
    vis_image = image.copy()

    for box, class_, score in zip(det_info.boxes, det_info.classes, det_info.scores):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(vis_image, p1, p2, color, thickness=2, lineType=cv2.LINE_AA)

        text = f"{class_} {score * 100:.1f}%"
        label_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

        x, y = int(box[0]), int(box[1] + 1)
        y = min(y, vis_image.shape[0])

        cv2.rectangle(vis_image, (x, y, label_size[0], label_size[1]), color, -1)
        cv2.putText(vis_image, text, (x, y + label_size[1] - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return vis_image