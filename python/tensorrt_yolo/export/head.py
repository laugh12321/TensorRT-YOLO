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
# Version :   6.2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/04/22 09:45:11
# Desc    :   YOLO Series Model head modules.
# ==============================================================================
import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, Value, nn
from ultralytics.nn.modules import (
    OBB,
    Classify,
    Conv,
    Detect,
    LRPCHead,
    Pose,
    Proto,
    Segment,
    WorldDetect,
    YOLOEDetect,
    YOLOESegment,
    v10Detect,
)
from ultralytics.nn.modules.conv import autopad
from ultralytics.utils.checks import check_version
from ultralytics.utils.tal import make_anchors

__all__ = [
    "YOLODetect",
    "YOLOSegment",
    "YOLOClassify",
    "YOLOV10Detect",
    "YOLOWorldDetect",
    "YOLOEDetectHead",
    "YOLOESegmentHead",
    "UltralyticsDetect",
    "UltralyticsOBB",
    "UltralyticsSegment",
    "UltralyticsPose",
    "UltralyticsClassify",
]


class EfficientNMS_TRT(torch.autograd.Function):
    """NMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: int = 100,
        score_activation: int = 1,
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
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: int = 100,
        score_activation: int = 1,
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientNMS_TRT',
            boxes,
            scores,
            outputs=4,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,  # 1, 将 sigmoid 激活应用于 NMS 操作中的置信度得分
            background_class_i=-1,  # 没有背景类别
            class_agnostic_i=1,  # 执行类无关的 NMS
            box_coding_i=1,  # 输入边框为 BoxCenterSize 格式 (x, y, w, h)
            plugin_version_s='1',  # 插件版本为 1
        )


class EfficientRotatedNMS_TRT(torch.autograd.Function):
    """RotatedNMS block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: int = 100,
        score_activation: int = 1,
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
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: int = 100,
        score_activation: int = 1,
    ) -> Tuple[Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientRotatedNMS_TRT',
            boxes,
            scores,
            outputs=4,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,  # 1, 将 sigmoid 激活应用于 NMS 操作中的置信度得分
            background_class_i=-1,  # 没有背景类别
            class_agnostic_i=1,  # 执行类无关的 NMS
            box_coding_i=1,  # 输入边框为 BoxCenterSize 格式 (x, y, w, h)
            plugin_version_s='1',  # 插件版本为 1
        )


class EfficientIdxNMS_TRT(torch.autograd.Function):
    """NMS with Index block for YOLO-fused model for TensorRT."""

    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: int = 100,
        score_activation: int = 1,
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
        score_threshold: float = 0.25,
        iou_threshold: float = 0.65,
        max_output_boxes: int = 100,
        score_activation: int = 1,
    ) -> Tuple[Value, Value, Value, Value, Value]:
        return g.op(
            'TRT::EfficientIdxNMS_TRT',
            boxes,
            scores,
            outputs=5,
            score_threshold_f=score_threshold,
            iou_threshold_f=iou_threshold,
            max_output_boxes_i=max_output_boxes,
            score_activation_i=score_activation,  # 1, 将 sigmoid 激活应用于 NMS 操作中的置信度得分
            background_class_i=-1,  # 没有背景类别
            class_agnostic_i=1,  # 执行类无关的 NMS
            box_coding_i=1,  # 输入边框为 BoxCenterSize 格式 (x, y, w, h)
            plugin_version_s='1',  # 插件版本为 1
        )


"""
===============================================================================
                         YOLOv3 and YOLOv5 Model heads
===============================================================================
"""


class YOLODetect(nn.Module):
    """YOLOv3 and YOLOv5 Detect head for detection models."""

    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

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
        box_lst = []
        conf_lst = []
        mask_lst = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

            if isinstance(self, YOLOSegment):  # (boxes + masks)
                xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh

                box_lst.append(torch.cat((xy, wh), 4).view(bs, self.na * nx * ny, -1))
                conf_lst.append(conf.sigmoid().view(bs, self.na * nx * ny, -1))
                mask_lst.append(mask.view(bs, self.na * nx * ny, -1))
            else:  # Detect (boxes only)
                xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh

                box_lst.append(torch.cat((xy, wh), 4).view(bs, self.na * nx * ny, -1))
                conf_lst.append(conf.view(bs, self.na * nx * ny, -1))

        boxes = torch.cat(box_lst, 1)  # (bs, na*nx*ny, 4)
        confs = torch.cat(conf_lst, 1)  # (bs, na*nx*ny, nc+1)
        scores = confs[..., 0:1] * confs[..., 1:]  # (bs, na*nx*ny, nc)

        if isinstance(self, YOLOSegment):  # (boxes + masks)
            return EfficientIdxNMS_TRT.apply(boxes, scores, self.conf_thres, self.iou_thres, self.max_det, 0), torch.cat(mask_lst, 1)
        else:  # Detect (boxes only)
            return EfficientNMS_TRT.apply(boxes, scores, self.conf_thres, self.iou_thres, self.max_det, 0)

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
    """YOLOv3 and YOLOv5 Segment head for segmentation models."""

    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

    def forward(self, x):
        p = self.proto(x[0])
        bs, _, mask_h, mask_w = p.shape
        (num_dets, det_boxes, det_scores, det_classes, det_indices), mc = YOLODetect.forward(self, x)

        # Retrieve the corresponding masks using batch and detection indices.
        bs_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1)
        selected_mc = mc[bs_indices, det_indices]
        det_masks = torch.einsum('b d n, b n h w -> b d h w', selected_mc, p).sigmoid()

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.uint8),
        )


class YOLOClassify(nn.Module):
    """YOLOv3 and YOLOv5 classification head."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)

        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return torch.stack(x.softmax(1).topk(min(5, x.shape[1]), largest=True, sorted=True), dim=-1)


"""
===============================================================================
                            Ultralytics Model heads
===============================================================================
"""


class BaseUltralyticsHead(nn.Module):
    """Base class for Ultralytics heads with common functionality."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

    def _new_inference(self, x):
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return dbox, cls


class UltralyticsDetect(Detect, BaseUltralyticsHead):
    """Ultralytics Detect head for detection models."""

    def forward(self, x):
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._new_inference(x)

        # Using transpose for compatibility with EfficientNMS_TRT
        return EfficientNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )


class UltralyticsOBB(OBB, BaseUltralyticsHead):
    """Ultralytics OBB detection head for rotated boxes."""

    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        self.angle = angle

        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._new_inference(x)
        rbox = torch.cat([dbox, angle], 1)

        # Using transpose for compatibility with EfficientRotatedNMS_TRT
        return EfficientRotatedNMS_TRT.apply(
            rbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )


class UltralyticsSegment(Segment, BaseUltralyticsHead):
    """Ultralytics Segment head for segmentation models."""

    def forward(self, x):
        p = self.proto(x[0])  # mask protos
        bs, _, mask_h, mask_w = p.shape
        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2).permute(0, 2, 1)  # mask coefficients

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._new_inference(x)

        ## Using transpose for compatibility with EfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )

        # Retrieve the corresponding masks using batch and detection indices.
        bs_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1)
        selected_mc = mc[bs_indices, det_indices]
        det_masks = torch.einsum('b d n, b n h w -> b d h w', selected_mc, p).sigmoid()

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.uint8),
        )


class UltralyticsPose(Pose, BaseUltralyticsHead):
    """Ultralytics Pose head for keypoint detection."""

    def forward(self, x):
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)

        # Detect forward
        x = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        dbox, cls = self._new_inference(x)

        ## Using transpose for compatibility with EfficientIdxNMS_TRT
        num_dets, det_boxes, det_scores, det_classes, det_indices = EfficientIdxNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )

        pred_kpts = self.kpts_decode(bs, kpt).transpose(1, 2)
        bs_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1)
        det_kpts = pred_kpts[bs_indices, det_indices].view(bs, self.max_det, *self.kpt_shape)

        return num_dets, det_boxes, det_scores, det_classes, det_kpts


class UltralyticsClassify(Classify):
    """Ultralytics classification head, i.e. x(b,c1,20,20) to x(b,min(5,cls_num),2)."""

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)

        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        return torch.stack(x.softmax(1).topk(min(5, x.shape[1]), largest=True, sorted=True), dim=-1)


class YOLOV10Detect(v10Detect):
    """YOLOv10 Detection head from https://arxiv.org/pdf/2405.14458."""

    max_det = 100
    iou_thres = 0.45
    conf_thres = 0.25

    def forward_end2end(self, x):
        x_detach = [xi.detach() for xi in x]
        one2one = [torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)]

        y = self._inference(one2one)
        return self.postprocess(y.permute(0, 2, 1), self.max_det, self.conf_thres, self.nc)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, conf_thres: float, nc: int = 80):
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        nums = (scores >= conf_thres).sum(dim=1, keepdim=True).int()
        return nums, boxes[i, index // nc], scores, (index % nc).to(torch.int32)


class YOLOWorldDetect(WorldDetect, BaseUltralyticsHead):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def forward(self, x, text):
        x = [torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1) for i in range(self.nl)]

        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        dbox, cls = self._new_inference(x)

        # Using transpose for compatibility with EfficientNMS_TRT
        return EfficientNMS_TRT.apply(
            dbox.transpose(1, 2),
            cls.transpose(1, 2),
            self.conf_thres,
            self.iou_thres,
            self.max_det,
        )


class YOLOEDetectHead(YOLOEDetect, BaseUltralyticsHead):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    segment = False

    def forward_lrpc(self, x, return_mask=False):
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](cls_feat, loc_feat, 0 if not self.dynamic else getattr(self, "conf", 0.001))
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)
        dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        mask = torch.cat(masks)
        if self.dynamic:
            dbox = dbox[..., mask]

        if return_mask:
            return EfficientIdxNMS_TRT.apply(
                dbox.transpose(1, 2),
                cls.transpose(1, 2),
                self.conf_thres,
                self.iou_thres,
                self.max_det,
            ), mask
        else:
            if self.segment:
                return EfficientIdxNMS_TRT.apply(
                    dbox.transpose(1, 2),
                    cls.transpose(1, 2),
                    self.conf_thres,
                    self.iou_thres,
                    self.max_det,
                )
            else:
                return EfficientNMS_TRT.apply(
                    dbox.transpose(1, 2),
                    cls.transpose(1, 2),
                    self.conf_thres,
                    self.iou_thres,
                    self.max_det,
                )

    def forward(self, x, cls_pe, return_mask=False):
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x, return_mask)

        x = [torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1) for i in range(self.nl)]

        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        dbox, cls = self._new_inference(x)

        if self.segment:
            return EfficientIdxNMS_TRT.apply(
                dbox.transpose(1, 2),
                cls.transpose(1, 2),
                self.conf_thres,
                self.iou_thres,
                self.max_det,
            )
        else:
            return EfficientNMS_TRT.apply(
                dbox.transpose(1, 2),
                cls.transpose(1, 2),
                self.conf_thres,
                self.iou_thres,
                self.max_det,
            )


class YOLOESegmentHead(YOLOESegment, YOLOEDetectHead):
    """YOLO segmentation head with text embedding capabilities."""

    segment = True

    def forward(self, x, text):
        p = self.proto(x[0])  # mask protos
        bs, _, mask_h, mask_w = p.shape
        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients

        if hasattr(self, "lrpc"):
            (num_dets, det_boxes, det_scores, det_classes, det_indices), mask = YOLOEDetectHead.forward(self, x, text, return_mask=True)
            mc = (mc * mask.int()) if not self.dynamic else mc[..., mask]
        else:
            num_dets, det_boxes, det_scores, det_classes, det_indices = YOLOEDetectHead.forward(self, x, text)

        mc = mc.permute(0, 2, 1)

        # Retrieve the corresponding masks using batch and detection indices.
        bs_indices = torch.arange(bs, device=det_classes.device, dtype=det_classes.dtype).unsqueeze(1)
        selected_mc = mc[bs_indices, det_indices]
        det_masks = torch.einsum('b d n, b n h w -> b d h w', selected_mc, p).sigmoid()

        return (
            num_dets,
            det_boxes,
            det_scores,
            det_classes,
            F.interpolate(det_masks, size=(mask_h * 4, mask_w * 4), mode="bilinear", align_corners=False).gt_(0.5).to(torch.uint8),
        )
