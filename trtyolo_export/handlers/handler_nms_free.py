#!/usr/bin/env python
# ==============================================================================
# Copyright (c) 2026 laugh12321 Authors. All Rights Reserved.
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
"""Handlers for modern NMS-free postprocess export graphs."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import numpy as np
import onnx_graphsurgeon as gs

from ..types import GraphContext, MatchResult, RewriteResult
from ..utils import get_static_dim, make_const_int64
from .handler_base import BaseHandler
from .handler_common import (
    build_nms_free_common_outputs,
    match_nms_free_obb_postprocess_inputs,
    match_nms_free_pose_postprocess_inputs,
    match_nms_free_segment_postprocess_inputs,
    reshape_pose_kpts,
    resolve_mask_protos,
    resolve_nms_free_detect_inputs,
    resolve_nms_free_obb_inputs,
    resolve_nms_free_pose_inputs,
    resolve_nms_free_segment_inputs,
    resolve_pose_kpts_shape_source,
)


class NmsFreePoseHandler(BaseHandler):
    """Handle NMS-free pose outputs."""

    name = "nms_free_pose"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match NMS-free pose postprocess graphs."""
        if match_nms_free_pose_postprocess_inputs(ctx.final_concat) is None:
            return None
        return MatchResult(score=95, reason="NMS-free pose postprocess pattern")

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose NMS-free pose detections and keypoints."""
        boxes_out, kpts_out, score_out, class_out = (
            resolve_nms_free_pose_inputs(ctx.final_concat)
        )
        boxes_out.name = "det_boxes"

        num_dets, det_scores, det_classes = build_nms_free_common_outputs(
            ctx.graph,
            score_out,
            class_out,
            score_thresh=ctx.score_thresh,
        )
        det_kpts = reshape_pose_kpts(
            ctx.graph, kpts_out, resolve_pose_kpts_shape_source(ctx.graph)
        )
        ctx.graph.outputs = [
            num_dets,
            boxes_out,
            det_scores,
            det_classes,
            det_kpts,
        ]
        return RewriteResult(handler_name=self.name, plugin_op=None)


class NmsFreeSegmentHandler(BaseHandler):
    """Handle NMS-free segmentation outputs."""

    name = "nms_free_segment"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match NMS-free segment postprocess graphs."""
        if ctx.mask_protos is None:
            return None
        mask_dim = get_static_dim(ctx.mask_protos.shape, 1)
        if (
            match_nms_free_segment_postprocess_inputs(
                ctx.final_concat, mask_dim
            )
            is None
        ):
            return None
        return MatchResult(
            score=96, reason="NMS-free segment postprocess pattern"
        )

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose NMS-free detections together with generated masks."""
        mask_protos = resolve_mask_protos(ctx)
        mask_dim = get_static_dim(mask_protos.shape, 1)
        if mask_dim is None:
            raise ValueError("Mask protos channel dim is not available")

        boxes_out, mask_coeffs, score_out, class_out = (
            resolve_nms_free_segment_inputs(ctx.final_concat, mask_dim)
        )
        boxes_out.name = "det_boxes"
        num_dets, det_scores, det_classes = build_nms_free_common_outputs(
            ctx.graph,
            score_out,
            class_out,
            score_thresh=ctx.score_thresh,
        )

        mask_coeffs_bnm = mask_coeffs
        coeff_dim_axis2 = get_static_dim(mask_coeffs.shape, 2)
        coeff_dim_axis1 = get_static_dim(mask_coeffs.shape, 1)
        if coeff_dim_axis2 != mask_dim and coeff_dim_axis1 == mask_dim:
            mask_coeffs_bnm = gs.Variable(
                name="mask_coeffs_bnm", dtype=mask_coeffs.dtype
            )
            ctx.graph.layer(
                op="Transpose",
                name="TransposeMaskCoeffNMSFree",
                inputs=[mask_coeffs],
                outputs=[mask_coeffs_bnm],
                attrs=OrderedDict(perm=[0, 2, 1]),
            )

        det_boxes_flat = gs.Variable(
            name="det_boxes_flat", dtype=boxes_out.dtype
        )
        ctx.graph.layer(
            op="Reshape",
            name="ReshapeDetBoxesFlat",
            inputs=[
                boxes_out,
                make_const_int64("det_boxes_flat_shape", [-1, 4]),
            ],
            outputs=[det_boxes_flat],
        )

        batch_static = get_static_dim(boxes_out.shape, 0)
        n_static = get_static_dim(boxes_out.shape, 1)
        if batch_static is None or n_static is None:
            raise ValueError(
                "NMS-free segment requires static det_boxes shape for mask ops"
            )

        batch_indices = make_const_int64(
            "det_batch_indices",
            np.repeat(np.arange(batch_static), n_static).tolist(),
        )

        mask_h = get_static_dim(mask_protos.shape, 2)
        mask_w = get_static_dim(mask_protos.shape, 3)
        if mask_h is None or mask_w is None:
            raise ValueError("Mask protos spatial dims are not available")

        pooled_proto = gs.Variable(name="pooled_proto", dtype=mask_protos.dtype)
        ctx.graph.layer(
            op="ROIAlign_TRT",
            name="ROIAlignMaskProtoNMSFree",
            inputs=[mask_protos, det_boxes_flat, batch_indices],
            outputs=[pooled_proto],
            attrs=OrderedDict(
                output_height=mask_h,
                output_width=mask_w,
                coordinate_transformation_mode=1,
                mode=1,
                sampling_ratio=0,
                spatial_scale=0.25,
            ),
        )

        selected_mc_flat = gs.Variable(
            name="selected_mask_coeffs_flat", dtype=mask_coeffs_bnm.dtype
        )
        ctx.graph.layer(
            op="Reshape",
            name="ReshapeSelectedMaskCoeffNMSFree",
            inputs=[
                mask_coeffs_bnm,
                make_const_int64("mask_coeffs_flat_shape", [-1, mask_dim]),
            ],
            outputs=[selected_mc_flat],
        )

        selected_mc_unsq = gs.Variable(
            name="selected_mask_coeffs_unsq", dtype=mask_coeffs_bnm.dtype
        )
        ctx.graph.layer(
            op="Unsqueeze",
            name="UnsqueezeSelectedMaskCoeffNMSFree",
            inputs=[
                selected_mc_flat,
                make_const_int64("mask_coeffs_unsq_axis", [1]),
            ],
            outputs=[selected_mc_unsq],
        )

        pooled_proto_flat = gs.Variable(
            name="pooled_proto_flat", dtype=mask_protos.dtype
        )
        mask_hw = mask_h * mask_w
        ctx.graph.layer(
            op="Reshape",
            name="ReshapePooledProtoNMSFree",
            inputs=[
                pooled_proto,
                make_const_int64(
                    "pooled_proto_flat_shape", [-1, mask_dim, mask_hw]
                ),
            ],
            outputs=[pooled_proto_flat],
        )

        mask_logits = gs.Variable(
            name="det_mask_logits", dtype=mask_protos.dtype
        )
        ctx.graph.layer(
            op="MatMul",
            name="MatMulMaskNMSFree",
            inputs=[selected_mc_unsq, pooled_proto_flat],
            outputs=[mask_logits],
        )

        mask_logits_sq = gs.Variable(
            name="det_mask_logits_sq", dtype=mask_protos.dtype
        )
        ctx.graph.layer(
            op="Squeeze",
            name="SqueezeMaskLogitsNMSFree",
            inputs=[mask_logits, make_const_int64("mask_logits_axis", [1])],
            outputs=[mask_logits_sq],
        )

        mask_sigmoid = gs.Variable(
            name="det_masks_sigmoid", dtype=mask_protos.dtype
        )
        ctx.graph.layer(
            op="Sigmoid",
            name="SigmoidDetMasksNMSFree",
            inputs=[mask_logits_sq],
            outputs=[mask_sigmoid],
        )

        det_masks = gs.Variable(name="det_masks", dtype=mask_protos.dtype)
        ctx.graph.layer(
            op="Reshape",
            name="ReshapeDetMasksNMSFree",
            inputs=[
                mask_sigmoid,
                make_const_int64(
                    "det_masks_shape", [batch_static, n_static, mask_h, mask_w]
                ),
            ],
            outputs=[det_masks],
        )

        ctx.graph.outputs = [
            num_dets,
            boxes_out,
            det_scores,
            det_classes,
            det_masks,
        ]
        return RewriteResult(handler_name=self.name, plugin_op=None)


class NmsFreeDetectHandler(BaseHandler):
    """Handle NMS-free detection outputs."""

    name = "nms_free_detect"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match NMS-free detect graphs using GatherElements/Unsqueeze/Cast."""
        concat = ctx.final_concat
        if len(concat.inputs) != 3:
            return None

        names = {inp.name or "" for inp in concat.inputs}
        keys = {"/GatherElements", "/Unsqueeze", "/Cast"}
        hits = {key for key in keys for name in names if key in name}
        if hits == keys:
            return MatchResult(
                score=90, reason="GatherElements+Unsqueeze+Cast inputs"
            )
        return None

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose NMS-free boxes, scores, and classes."""
        boxes_out, score_out, class_out = resolve_nms_free_detect_inputs(
            ctx.final_concat
        )
        boxes_out.name = "det_boxes"

        num_dets, det_scores, det_classes = build_nms_free_common_outputs(
            ctx.graph,
            score_out,
            class_out,
            score_thresh=ctx.score_thresh,
        )
        ctx.graph.outputs = [num_dets, boxes_out, det_scores, det_classes]
        return RewriteResult(handler_name=self.name, plugin_op=None)


class NmsFreeObbHandler(BaseHandler):
    """Handle NMS-free oriented bounding box outputs."""

    name = "nms_free_obb"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match NMS-free OBB postprocess graphs."""
        if match_nms_free_obb_postprocess_inputs(ctx.final_concat) is None:
            return None
        return MatchResult(score=95, reason="NMS-free OBB postprocess pattern")

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose NMS-free rotated boxes, scores, and classes."""
        boxes_out, score_out, class_out = resolve_nms_free_obb_inputs(
            ctx.graph, ctx.final_concat
        )
        num_dets, det_scores, det_classes = build_nms_free_common_outputs(
            ctx.graph,
            score_out,
            class_out,
            score_thresh=ctx.score_thresh,
        )
        ctx.graph.outputs = [num_dets, boxes_out, det_scores, det_classes]
        return RewriteResult(handler_name=self.name, plugin_op=None)
