#!/usr/bin/env python
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
# File    :   head.py
# Version :   2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/04/22 09:45:11
# Desc    :   YOLO Series Detect Head.
# ==============================================================================
import math
from typing import Tuple

import torch
from torch import Tensor, Value, nn
from ultralytics.nn.modules import DFL, Conv
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import dist2bbox, make_anchors

__all__ = ["YOLODetect", "UltralyticsDetect"]


class Efficient_TRT_NMS(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        max_output_boxes: float = 100,
        box_coding: int = 1,
        background_class: int = -1,
        score_activation: int = 0,
        plugin_version: str = '1',
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_detections = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        detection_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        detection_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        detection_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_detections, detection_boxes, detection_scores, detection_classes

    @staticmethod
    def symbolic(
        g,
        boxes,
        scores,
        iou_threshold: float = 0.65,
        score_threshold: float = 0.25,
        max_output_boxes: float = 100,
        box_coding: int = 1,
        background_class: int = -1,
        score_activation: int = 0,
        plugin_version: str = '1',
    ) -> Tuple[Value, Value, Value, Value]:
        out = g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            outputs=4,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=background_class,
            score_activation_i=score_activation,
            plugin_version_s=plugin_version,
        )
        num_detections, detection_boxes, detection_scores, detection_classes = out
        return num_detections, detection_boxes, detection_scores, detection_classes


"""
===============================================================================
            YOLOv3 and YOLOv5 Detect head for detection models
===============================================================================
"""


class YOLODetect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    iou_thres = 0.65
    conf_thres = 0.25
    max_det = 100

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        boxes, scores = [], []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh

            box = torch.cat((xy, wh), 4)
            boxes.append(box.view(bs, self.na * nx * ny, 4))
            score = conf[..., 0].unsqueeze(-1) * conf[..., 1:]
            scores.append(score.view(bs, self.na * nx * ny, self.nc))

        return Efficient_TRT_NMS.apply(
            torch.cat(boxes, 1),
            torch.cat(scores, 1),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


"""
===============================================================================
            Ultralytics Detect head for detection models
===============================================================================
"""


class UltralyticsDetect(nn.Module):
    """YOLOv8 Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    max_det = 100
    iou_thres = 0.65
    conf_thres = 0.25
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLOv8 detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export:
            return Efficient_TRT_NMS.apply(
                torch.transpose(dbox, 1, 2),
                torch.transpose(cls.sigmoid(), 1, 2),
                self.iou_thres,
                self.conf_thres,
                self.max_det,
            )
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
