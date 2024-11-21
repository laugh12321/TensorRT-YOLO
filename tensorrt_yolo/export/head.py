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
# File    :   head.py
# Version :   5.0.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/04/22 09:45:11
# Desc    :   YOLO Series Model head modules.
# ==============================================================================
import copy
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, Value, nn
from ultralytics.nn.modules import OBB, Conv, Detect, Pose, Proto, Segment
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import make_anchors

__all__ = ["YOLODetect", "YOLOSegment", "V10Detect", "UltralyticsDetect", "UltralyticsOBB", "UltralyticsSegment", "UltralyticsPose"]  # noqa: F822


class EfficientNMS_TRT(torch.autograd.Function):
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
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes

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
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
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
            class_agnostic_i=class_agnostic,
            plugin_version_s=plugin_version,
        )


class EfficientRotatedNMS_TRT(torch.autograd.Function):
    """RotatedNMS block for YOLO-fused model for TensorRT."""

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
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 5, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes

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
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientRotatedNMS_TRT',
            boxes,
            scores,
            outputs=4,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=background_class,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            plugin_version_s=plugin_version,
        )


class EfficientIdxNMS_TRT(torch.autograd.Function):
    """NMS with Index block for YOLO-fused model for TensorRT."""

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
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, num_boxes, num_classes = scores.shape
        num_dets = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4, dtype=torch.float32)
        det_scores = torch.randn(batch_size, max_output_boxes, dtype=torch.float32)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        det_indices = torch.randint(0, num_boxes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_dets, det_boxes, det_scores, det_classes, det_indices

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
        class_agnostic: int = 1,
        plugin_version: str = '1',
    ) -> Tuple[Value, Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientIdxNMS_TRT',
            boxes,
            scores,
            outputs=5,
            box_coding_i=box_coding,
            iou_threshold_f=iou_threshold,
            score_threshold_f=score_threshold,
            max_output_boxes_i=max_output_boxes,
            background_class_i=background_class,
            score_activation_i=score_activation,
            class_agnostic_i=class_agnostic,
            plugin_version_s=plugin_version,
        )


"""
===============================================================================
        YOLOv3 and YOLOv5 Model head for detection and segmentation models
===============================================================================
"""


class YOLODetect(nn.Module):
    """YOLOv3 and YOLOv5 Detect head for detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    iou_thres = 0.45
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
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs, 255, 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2)

            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
            xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))

        z = torch.cat(z, 1)

        # Separate boxes and scores for EfficientNMS_TRT
        boxes, conf = z[..., :4], z[..., 4:]
        scores = conf[..., 0:1] * conf[..., 1:]

        return EfficientNMS_TRT.apply(boxes, scores, self.iou_thres, self.conf_thres, self.max_det)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class YOLOSegment(YOLODetect):
    """YOLOv3 and YOLOv5 Segment head for segmentation models, extending Detect with mask and prototype layers."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    iou_thres = 0.45
    conf_thres = 0.25
    max_det = 100

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv3 and YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        bs, _, mask_h, mask_w = p.shape

        # Detect forward
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs, 255, 20, 20) to x(bs, 3, 20, 20, 85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2)

            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
            xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
            wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
            y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
            z.append(y.view(bs, self.na * nx * ny, self.no))

        z = torch.cat(z, 1)

        # Separate boxes and scores for EfficientIdxNMS_TRT
        boxes, conf, mc = z[..., :4], z[..., 4 : self.no - self.nm], z[..., self.no - self.nm :]
        scores = conf[..., 0:1] * conf[..., 1:]
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            boxes,
            scores,
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

        # Retrieve the corresponding masks using batch and detection indices.
        batch_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1).expand(-1, self.max_det).view(-1)
        det_indices = det_indices.view(-1)
        selected_masks = mc[batch_indices, det_indices].view(bs, self.max_det, 1, self.nm)

        masks_protos = p.view(bs, self.nm, mask_h * mask_w)
        det_masks = torch.matmul(selected_masks, masks_protos).sigmoid().view(bs, self.max_det, mask_h, mask_w)

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.uint8),
        )


"""
===============================================================================
        Ultralytics Model head for detection and segmentation models
===============================================================================
"""


class UltralyticsDetect(Detect):
    """Ultralytics Detect head for detection models."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        dbox, cls = self._inference(x)

        # Using transpose for compatibility with EfficientNMS_TRT
        return EfficientNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

    def forward_end2end(self, x):
        """Performs forward pass of the v10Detect module."""
        x_detach = [xi.detach() for xi in x]
        one2one = [torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)]

        dbox, cls = self._inference(one2one)
        y = torch.cat((dbox, cls), 1)
        det_boxes, det_scores, det_classes = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        num_dets = (det_scores >= self.conf_thres).sum(dim=1, keepdim=True).int()
        return num_dets, det_boxes, det_scores, det_classes

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Processed predictions with shapes:
                - boxes (torch.Tensor): Shape (batch_size * min(max_det, num_anchors), 4) with last dimension
                    format [x, y, w, h].
                - scores (torch.Tensor): Shape (batch_size * min(max_det, anchors),) with values representing
                    the max class probability for each detection.
                - class_index (torch.Tensor): Shape (batch_size * min(max_det, anchors),) with values representing
                    the index of the class with the max probability for each detection.
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return boxes[i, index // nc], scores, (index % nc).to(torch.int32)


class UltralyticsOBB(OBB):
    """Ultralytics OBB detection head for detection with rotation models."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        self.angle = angle

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        rotated_box = torch.cat([dbox.transpose(1, 2), angle.transpose(1, 2)], 2)

        # Using transpose for compatibility with EfficientRotatedNMS_TRT
        return EfficientRotatedNMS_TRT.apply(
            rotated_box,
            cls.sigmoid().transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )


class UltralyticsSegment(Segment):
    """Ultralytics Segment head for segmentation models."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs, _, mask_h, mask_w = p.shape
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2).permute(0, 2, 1)  # mask coefficients

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._inference(x)

        ## Using transpose for compatibility with EfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

        # Retrieve the corresponding masks using batch and detection indices.
        batch_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1).expand(-1, self.max_det).view(-1)
        det_indices = det_indices.view(-1)
        selected_masks = mc[batch_indices, det_indices].view(bs, self.max_det, 1, self.nm)

        masks_protos = p.view(bs, self.nm, mask_h * mask_w)
        det_masks = torch.matmul(selected_masks, masks_protos).sigmoid().view(bs, self.max_det, mask_h, mask_w)

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.uint8),
        )

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()


class UltralyticsPose(Pose):
    """Ultralytics Pose head for keypoints models."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._inference(x)

        ## Using transpose for compatibility with EfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.iou_thres,
            self.conf_thres,
            self.max_det,
        )

        batch_indices = (
            torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1).expand(-1, self.max_det).reshape(-1)
        )
        det_indices = det_indices.view(-1)
        pred_kpt = self.kpts_decode(bs, kpt)

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            pred_kpt[batch_indices, det_indices].view(bs, self.max_det, self.kpt_shape[0], self.kpt_shape[1]),
        )

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls.sigmoid()

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        # NCNN fix
        y = kpts.view(bs, *self.kpt_shape, -1)
        a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
        if ndim == 3:
            a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
        return a.view(bs, self.nk, -1).transpose(1, 2)


class v10Detect(UltralyticsDetect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
