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
# File    :   nms.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/28 13:35:46
# Desc    :   YOLOv8 NMS-fused detection model for TensorRT export.
# ==============================================================================
from typing import Tuple

import torch
import torch.nn as nn
from torch import Graph, Tensor, Value

from ultralytics.utils.tal import make_anchors

__all__ = ['PostDetectTRTNMS']


class Efficient_TRT_NMS(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx: Graph,
        boxes: Tensor,
        scores: Tensor,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        max_output_boxes: int = 100,
        background_class: int = -1,
        box_coding: int = 0,
        plugin_version: str = '1',
        score_activation: int = 0,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0,
                                 max_output_boxes, (batch_size, 1),
                                 dtype=torch.int32)
        boxes = torch.randn(batch_size, max_output_boxes, 4)
        scores = torch.randn(batch_size, max_output_boxes)
        labels = torch.randint(0,
                               num_classes, (batch_size, max_output_boxes),
                               dtype=torch.int32)

        return num_dets, boxes, scores, labels

    @staticmethod
    def symbolic(
        g,
        boxes: Value,
        scores: Value,
        iou_threshold: float = 0.45,
        score_threshold: float = 0.25,
        max_output_boxes: int = 100,
        background_class: int = -1,
        box_coding: int = 0,
        score_activation: int = 0,
        plugin_version: str = '1',
    ) -> Tuple[Value, Value, Value, Value]:
        out = g.op('TRT::EfficientNMS_TRT',
                   boxes,
                   scores,
                   iou_threshold_f=iou_threshold,
                   score_threshold_f=score_threshold,
                   max_output_boxes_i=max_output_boxes,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   plugin_version_s=plugin_version,
                   score_activation_i=score_activation,
                   outputs=4,
                   )
        nums_dets, boxes, scores, classes = out
        return nums_dets, boxes, scores, classes


class PostDetectTRTNMS(nn.Module):
    """YOLOv8 NMS-fused detection model for TensorRT export."""
    export = True
    shape = None
    dynamic = False
    iou_thres = 0.65
    conf_thres = 0.25
    max_det = 100

    def _forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        '''Decode yolov8 model output'''
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        res = x
        b, b_reg_num = shape[0], self.reg_max * 4
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(
                0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape
        x = [i.view(b, self.no, -1) for i in res]
        y = torch.cat(x, 2)
        boxes, scores = y[:, :b_reg_num, ...], y[:, b_reg_num:, ...].sigmoid()
        boxes = boxes.view(b, 4, self.reg_max, -1).permute(0, 1, 3, 2)
        boxes = boxes.softmax(-1) @ torch.arange(self.reg_max, device=boxes.device, dtype=boxes.dtype)
        boxes0, boxes1 = -boxes[:, :2, ...], boxes[:, 2:, ...]
        boxes = self.anchors.repeat(b, 2, 1) + torch.cat([boxes0, boxes1], 1)
        boxes = boxes * self.strides

        # output shape (bs, 4, spatial_dim), (bs, num_classes, spatial_dim)
        return boxes, scores

    def forward(self, x):
        boxes, scores = self._forward(x)

        return Efficient_TRT_NMS.apply(
            boxes.transpose(1, 2),
            scores.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det)
