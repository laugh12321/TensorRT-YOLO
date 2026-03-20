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
"""Handlers for legacy YOLO export layouts.

These handlers reshape outputs into TensorRT-friendly forms.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

import onnx_graphsurgeon as gs

from ..types import GraphContext, MatchResult, RewriteResult
from ..utils import (
    get_static_dim,
    infer_num_classes_from_concat,
    slice_axis_range,
    transpose_bnc_to_bcn,
)
from .handler_base import BaseHandler
from .handler_common import (
    infer_segment_num_classes,
    match_legacy_obb_head_inputs,
    resolve_mask_protos,
    resolve_segment_feature_axis,
)


class LegacyDetectUltralyticsHandler(BaseHandler):
    """Handle legacy Ultralytics detect heads with Mul and Sigmoid outputs."""

    name = "legacy_detect_ultralytics"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match a legacy detect Concat built from Mul and Sigmoid inputs."""
        concat = ctx.final_concat
        if concat.op != "Concat":
            return None
        if len(concat.inputs) != 2:
            return None

        prod0 = concat.i(0)
        prod1 = concat.i(1)
        has_sigmoid = (prod0 is not None and prod0.op == "Sigmoid") or (
            prod1 is not None and prod1.op == "Sigmoid"
        )
        has_mul = (prod0 is not None and prod0.op == "Mul") or (
            prod1 is not None and prod1.op == "Mul"
        )
        score = 95 if has_sigmoid and has_mul else 60
        reason = (
            "Concat has 2 inputs (Sigmoid/Mul)"
            if score == 95
            else "Concat has 2 inputs"
        )
        return MatchResult(score=score, reason=reason)

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose legacy detect boxes and scores as TensorRT-ready outputs."""
        concat = ctx.final_concat
        in0, in1 = concat.inputs
        prod0 = concat.i(0)
        prod1 = concat.i(1)
        if (
            prod0 is not None
            and prod0.op == "Sigmoid"
            and prod1 is not None
            and prod1.op == "Mul"
        ):
            sigmoid_in, mul_in = in0, in1
        elif (
            prod1 is not None
            and prod1.op == "Sigmoid"
            and prod0 is not None
            and prod0.op == "Mul"
        ):
            sigmoid_in, mul_in = in1, in0
        else:
            raise ValueError("Invalid input ops for legacy detect Concat")

        boxes_bcn = transpose_bnc_to_bcn(ctx.graph, mul_in)
        scores_bcn = transpose_bnc_to_bcn(ctx.graph, sigmoid_in)
        ctx.graph.outputs = [boxes_bcn, scores_bcn]
        return RewriteResult(
            handler_name=self.name, plugin_op="EfficientNMS_TRT"
        )


class LegacyObbUltralyticsHandler(BaseHandler):
    """Handle legacy Ultralytics OBB heads."""

    name = "legacy_obb_ultralytics"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match the legacy OBB concat pattern."""
        if match_legacy_obb_head_inputs(ctx.final_concat) is None:
            return None
        return MatchResult(
            score=95, reason="Legacy OBB pattern (Mul/Mul/Sigmoid)"
        )

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Assemble rotated boxes and scores for TensorRT OBB plugins."""
        matched = match_legacy_obb_head_inputs(ctx.final_concat)
        if matched is None:
            raise ValueError(
                "Matched OBB concat case but failed to resolve OBB inputs"
            )

        dbox_in, angle_in, sigmoid_in = matched
        rbox_concat = gs.Variable(name="rbox_output", dtype=dbox_in.dtype)
        ctx.graph.layer(
            op="Concat",
            name="ConcatRBox",
            inputs=[dbox_in, angle_in],
            outputs=[rbox_concat],
            attrs=OrderedDict(axis=1),
        )

        rbox_bcn = transpose_bnc_to_bcn(ctx.graph, rbox_concat)
        scores_bcn = transpose_bnc_to_bcn(ctx.graph, sigmoid_in)
        ctx.graph.outputs = [rbox_bcn, scores_bcn]
        return RewriteResult(
            handler_name=self.name, plugin_op="EfficientRotatedNMS_TRT"
        )


class LegacyPoseUltralyticsHandler(BaseHandler):
    """Handle legacy Ultralytics pose heads."""

    name = "legacy_pose_ultralytics"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match pose heads built from Mul, Sigmoid, and Reshape inputs."""
        concat = ctx.final_concat
        if concat.op != "Concat":
            return None
        if len(concat.inputs) != 3:
            return None

        mul_in = None
        sigmoid_in = None
        reshape_in = None
        for idx, input_tensor in enumerate(concat.inputs):
            producer = concat.i(idx)
            if producer is None:
                return None
            if producer.op == "Mul":
                mul_in = input_tensor
            elif producer.op == "Sigmoid":
                sigmoid_in = input_tensor
            elif producer.op == "Reshape":
                reshape_in = input_tensor
            else:
                return None

        if mul_in is None or sigmoid_in is None or reshape_in is None:
            return None

        if (
            get_static_dim(mul_in.shape, 1) != 4
            or get_static_dim(sigmoid_in.shape, 1) != 1
        ):
            return None
        return MatchResult(
            score=95, reason="Legacy pose pattern (Mul=4, Sigmoid=1, Reshape)"
        )

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose boxes, scores, and keypoints for pose postprocessing."""
        concat = ctx.final_concat
        mul_in = None
        sigmoid_in = None
        reshape_node = None

        for idx, input_tensor in enumerate(concat.inputs):
            producer = concat.i(idx)
            if producer is None:
                continue
            if producer.op == "Mul":
                mul_in = input_tensor
            elif producer.op == "Sigmoid":
                sigmoid_in = input_tensor
            elif producer.op == "Reshape":
                reshape_node = producer

        if mul_in is None or sigmoid_in is None or reshape_node is None:
            raise ValueError("Invalid input ops for legacy pose Concat")
        if len(reshape_node.inputs) == 0:
            raise ValueError("Reshape node has no inputs for legacy pose")

        boxes_bcn = transpose_bnc_to_bcn(ctx.graph, mul_in)
        scores_bcn = transpose_bnc_to_bcn(ctx.graph, sigmoid_in)
        ctx.graph.outputs = [boxes_bcn, scores_bcn, reshape_node.inputs[0]]
        return RewriteResult(
            handler_name=self.name, plugin_op="EfficientIdxNMS_TRT"
        )


class UltralyticsSegmentHandler(BaseHandler):
    """Handle Ultralytics segmentation heads with mask prototypes."""

    name = "segment_ultralytics"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match segment heads with box, score, mask, and proto branches."""
        if ctx.mask_protos is None:
            return None
        concat = ctx.final_concat
        if concat.op != "Concat":
            return None
        if len(concat.inputs) != 3:
            return None

        prod_ops = {
            concat.i(idx).op for idx in range(3) if concat.i(idx) is not None
        }
        if {"Mul", "Sigmoid", "Concat"} - prod_ops:
            return None
        return MatchResult(
            score=96,
            reason="Ultralytics segment pattern (Mul/Sigmoid/Concat + protos)",
        )

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Expose segmentation outputs expected by TensorRT-YOLO."""
        concat = ctx.final_concat
        mul_in = None
        sigmoid_in = None
        mask_coeffs = None

        for idx, input_tensor in enumerate(concat.inputs):
            producer = concat.i(idx)
            if producer is None:
                continue
            if producer.op == "Mul":
                mul_in = input_tensor
            elif producer.op == "Sigmoid":
                sigmoid_in = input_tensor
            elif producer.op == "Concat":
                mask_coeffs = input_tensor

        if mul_in is None or sigmoid_in is None or mask_coeffs is None:
            raise ValueError("Invalid input ops for ultralytics segment Concat")

        mask_protos = resolve_mask_protos(ctx)
        if get_static_dim(mask_protos.shape, 1) is None:
            raise ValueError("Mask protos channel dim is not available")

        boxes_bcn = transpose_bnc_to_bcn(ctx.graph, mul_in)
        scores_bcn = transpose_bnc_to_bcn(ctx.graph, sigmoid_in)
        ctx.graph.outputs = [boxes_bcn, scores_bcn, mask_coeffs, mask_protos]
        return RewriteResult(
            handler_name=self.name, plugin_op="EfficientIdxNMS_TRT"
        )


class LegacySegmentV3V5Handler(BaseHandler):
    """Handle legacy YOLOv3/YOLOv5 segmentation heads."""

    name = "legacy_segment_v3v5"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match legacy segment heads exported with view or reshape tensors."""
        if ctx.mask_protos is None:
            return None
        if ctx.final_concat.op != "Concat":
            return None
        names = {inp.name or "" for inp in ctx.final_concat.inputs}
        if len(ctx.final_concat.inputs) == 3 and (
            all("view_" in n for n in names)
            or all("/Reshape_" in n for n in names)
        ):
            return MatchResult(
                score=90,
                reason="Legacy v3/v5 segment pattern (view_/Reshape_ + protos)",
            )
        return None

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Split legacy segment outputs.

        Produces boxes, scores, mask coefficients, and prototype tensors.
        """
        concat_output = ctx.concat_output
        mask_protos = resolve_mask_protos(ctx)
        mask_dim = get_static_dim(mask_protos.shape, 1)
        if mask_dim is None:
            raise ValueError("Mask protos channel dim is not available")

        feature_axis = resolve_segment_feature_axis(
            concat_output, base_dim=5, mask_dim=mask_dim
        )
        num_classes = infer_segment_num_classes(
            ctx.graph,
            concat_output,
            mask_dim=mask_dim,
            base_dim=5,
            feature_axis=feature_axis,
        )

        boxes_xywh = slice_axis_range(
            ctx.graph,
            concat_output,
            0,
            4,
            feature_axis,
            "SliceBoxes",
            "boxes_output",
        )
        obj_score = slice_axis_range(
            ctx.graph,
            concat_output,
            4,
            5,
            feature_axis,
            "SliceObjScore",
            "obj_score_output",
        )
        class_score = slice_axis_range(
            ctx.graph,
            concat_output,
            5,
            5 + num_classes,
            feature_axis,
            "SliceClassesScore",
            "class_score_output",
        )
        mask_coeffs = slice_axis_range(
            ctx.graph,
            concat_output,
            5 + num_classes,
            5 + num_classes + mask_dim,
            feature_axis,
            "SliceMaskCoeff",
            "mask_coeffs_output",
        )

        combined_score = gs.Variable(
            name="score_output", dtype=concat_output.dtype
        )
        ctx.graph.layer(
            op="Mul",
            name="Mul_score",
            inputs=[obj_score, class_score],
            outputs=[combined_score],
        )
        ctx.graph.outputs = [
            boxes_xywh,
            combined_score,
            mask_coeffs,
            mask_protos,
        ]
        return RewriteResult(
            handler_name=self.name, plugin_op="EfficientIdxNMS_TRT"
        )


class LegacyDetectV3V5Handler(BaseHandler):
    """Handle legacy YOLOv3/YOLOv5 detection heads."""

    name = "legacy_detect_v3v5"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match legacy detect heads exported with view or reshape tensors."""
        if ctx.final_concat.op != "Concat":
            return None
        names = {inp.name or "" for inp in ctx.final_concat.inputs}
        if len(ctx.final_concat.inputs) == 3 and (
            all("view_" in n for n in names)
            or all("/Reshape_" in n for n in names)
        ):
            return MatchResult(
                score=85, reason="Concat inputs are all view_/Reshape_"
            )
        return None

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Split legacy detect outputs into boxes and combined scores."""
        concat_output = ctx.concat_output
        num_classes = infer_num_classes_from_concat(
            ctx.graph, concat_output.shape
        )

        boxes_xywh = slice_axis_range(
            ctx.graph, concat_output, 0, 4, 2, "SliceBoxes", "boxes_output"
        )
        obj_score = slice_axis_range(
            ctx.graph,
            concat_output,
            4,
            5,
            2,
            "SliceObjScore",
            "obj_score_output",
        )
        class_score = slice_axis_range(
            ctx.graph,
            concat_output,
            5,
            5 + num_classes,
            2,
            "SliceClassesScore",
            "class_score_output",
        )

        combined_score = gs.Variable(
            name="score_output", dtype=concat_output.dtype
        )
        ctx.graph.layer(
            op="Mul",
            name="Mul_score",
            inputs=[obj_score, class_score],
            outputs=[combined_score],
        )
        ctx.graph.outputs = [boxes_xywh, combined_score]
        return RewriteResult(
            handler_name=self.name, plugin_op="EfficientNMS_TRT"
        )
