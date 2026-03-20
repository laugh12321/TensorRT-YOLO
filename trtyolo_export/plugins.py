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
"""Helpers for appending TensorRT plugin nodes to rewritten graphs."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import onnx_graphsurgeon as gs
from onnx import TensorProto

from .utils import get_static_dim, make_const_int64


def append_trt_nms_plugin(
    onnx_graph: gs.Graph,
    is_dynamic: bool,
    batch_size: int,
    max_dets: int,
    score_thresh: float,
    iou_thresh: float,
    plugin_op: str,
) -> None:
    """Append a TensorRT NMS plugin and replace graph outputs."""
    is_rotated = plugin_op == "EfficientRotatedNMS_TRT"
    is_idx = plugin_op == "EfficientIdxNMS_TRT"
    box_dim = 5 if is_rotated else 4

    nms_inputs = list(onnx_graph.outputs)
    kpts_input = None
    mask_coeffs = None
    mask_protos = None
    is_segment = False
    if is_idx:
        if len(nms_inputs) < 3:
            raise ValueError(
                "EfficientIdxNMS_TRT requires [boxes, scores, kpts] or "
                "[boxes, scores, mask_coeffs, mask_protos]"
            )
        if len(nms_inputs) >= 4:
            is_segment = True
            mask_coeffs = nms_inputs[2]
            mask_protos = nms_inputs[3]
        else:
            kpts_input = nms_inputs[2]
        nms_inputs = nms_inputs[:2]

    op_outputs = [
        gs.Variable(
            name="num_dets",
            dtype=np.int32,
            shape=["batch" if is_dynamic else batch_size, 1],
        ),
        gs.Variable(
            name="det_boxes",
            dtype=np.float32,
            shape=["batch" if is_dynamic else batch_size, max_dets, box_dim],
        ),
        gs.Variable(
            name="det_scores",
            dtype=np.float32,
            shape=["batch" if is_dynamic else batch_size, max_dets],
        ),
        gs.Variable(
            name="det_classes",
            dtype=np.int32,
            shape=["batch" if is_dynamic else batch_size, max_dets],
        ),
    ]

    if is_idx:
        op_outputs.append(
            gs.Variable(
                name="det_indices",
                dtype=np.int32,
                shape=["batch" if is_dynamic else batch_size, max_dets],
            )
        )

    attrs = OrderedDict(
        plugin_version="1",
        background_class=-1,
        max_output_boxes=max_dets,
        score_threshold=score_thresh,
        iou_threshold=iou_thresh,
        score_activation=False,
        class_agnostic=False,
        box_coding=True,
    )

    onnx_graph.layer(
        op=plugin_op,
        name=plugin_op,
        inputs=nms_inputs,
        outputs=op_outputs,
        attrs=attrs,
    )

    if not is_idx:
        onnx_graph.outputs = op_outputs
        return

    det_indices = op_outputs[-1]
    indices_int64 = gs.Variable(
        name="det_indices_int64", dtype=TensorProto.INT64
    )
    onnx_graph.layer(
        op="Cast",
        name="CastDetIndicesToInt64",
        inputs=[det_indices],
        outputs=[indices_int64],
        attrs={"to": TensorProto.INT64},
    )

    indices_expanded = gs.Variable(
        name="det_indices_expanded", dtype=TensorProto.INT64
    )
    onnx_graph.layer(
        op="Unsqueeze",
        name="UnsqueezeDetIndices",
        inputs=[indices_int64, make_const_int64("det_indices_axis", [2])],
        outputs=[indices_expanded],
    )

    if is_segment:
        if mask_coeffs is None or mask_protos is None:
            raise ValueError(
                "Segment NMS requires mask coeffs and protos outputs"
            )

        mask_dim = get_static_dim(mask_protos.shape, 1)
        if mask_dim is None:
            raise ValueError("Mask protos channel dim is not available")

        mask_coeffs_bnm = mask_coeffs
        coeff_dim_axis2 = get_static_dim(mask_coeffs.shape, 2)
        coeff_dim_axis1 = get_static_dim(mask_coeffs.shape, 1)
        if coeff_dim_axis2 != mask_dim and coeff_dim_axis1 == mask_dim:
            mask_coeffs_bnm = gs.Variable(
                name="mask_coeffs_bnm", dtype=mask_coeffs.dtype
            )
            onnx_graph.layer(
                op="Transpose",
                name="TransposeMaskCoeff",
                inputs=[mask_coeffs],
                outputs=[mask_coeffs_bnm],
                attrs=OrderedDict(perm=[0, 2, 1]),
            )

        selected_mc = gs.Variable(
            name="selected_mask_coeffs", dtype=mask_coeffs_bnm.dtype
        )
        onnx_graph.layer(
            op="GatherND",
            name="GatherMaskCoeff",
            inputs=[mask_coeffs_bnm, indices_expanded],
            outputs=[selected_mc],
            attrs={"batch_dims": 1},
        )

        det_boxes = op_outputs[1]
        det_boxes_flat = gs.Variable(
            name="det_boxes_flat", dtype=det_boxes.dtype
        )
        onnx_graph.layer(
            op="Reshape",
            name="ReshapeDetBoxesFlat",
            inputs=[
                det_boxes,
                make_const_int64("det_boxes_flat_shape", [-1, 4]),
            ],
            outputs=[det_boxes_flat],
        )

        det_boxes_shape = gs.Variable(
            name="det_boxes_shape", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Shape",
            name="ShapeDetBoxes",
            inputs=[det_boxes],
            outputs=[det_boxes_shape],
        )

        batch_dim = gs.Variable(
            name="det_boxes_batch_dim", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Slice",
            name="SliceDetBoxesBatch",
            inputs=[
                det_boxes_shape,
                make_const_int64("det_boxes_batch_start", [0]),
                make_const_int64("det_boxes_batch_end", [1]),
                make_const_int64("det_boxes_batch_axes", [0]),
            ],
            outputs=[batch_dim],
        )

        batch_dim_scalar = gs.Variable(
            name="det_boxes_batch_dim_scalar", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Squeeze",
            name="SqueezeDetBoxesBatch",
            inputs=[batch_dim, make_const_int64("det_boxes_batch_axis", [0])],
            outputs=[batch_dim_scalar],
        )

        batch_range = gs.Variable(
            name="det_batch_range", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Range",
            name="RangeDetBatch",
            inputs=[
                make_const_int64("det_batch_range_start", [0]),
                batch_dim_scalar,
                make_const_int64("det_batch_range_step", [1]),
            ],
            outputs=[batch_range],
        )

        batch_range_unsq = gs.Variable(
            name="det_batch_range_unsq", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Unsqueeze",
            name="UnsqueezeDetBatchRange",
            inputs=[batch_range, make_const_int64("det_batch_unsq_axis", [1])],
            outputs=[batch_range_unsq],
        )

        batch_tiled = gs.Variable(
            name="det_batch_tiled", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Tile",
            name="TileDetBatch",
            inputs=[
                batch_range_unsq,
                make_const_int64("det_batch_repeats", [1, max_dets]),
            ],
            outputs=[batch_tiled],
        )

        batch_indices = gs.Variable(
            name="det_batch_indices", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Reshape",
            name="ReshapeDetBatchIndices",
            inputs=[
                batch_tiled,
                make_const_int64("det_batch_flat_shape", [-1]),
            ],
            outputs=[batch_indices],
        )

        mask_h = get_static_dim(mask_protos.shape, 2)
        mask_w = get_static_dim(mask_protos.shape, 3)
        if mask_h is None or mask_w is None:
            raise ValueError("Mask protos spatial dims are not available")

        pooled_proto = gs.Variable(name="pooled_proto", dtype=mask_protos.dtype)
        onnx_graph.layer(
            op="ROIAlign_TRT",
            name="ROIAlignMaskProto",
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
        onnx_graph.layer(
            op="Reshape",
            name="ReshapeSelectedMaskCoeff",
            inputs=[
                selected_mc,
                make_const_int64("mask_coeffs_flat_shape", [-1, mask_dim]),
            ],
            outputs=[selected_mc_flat],
        )

        selected_mc_unsq = gs.Variable(
            name="selected_mask_coeffs_unsq", dtype=mask_coeffs_bnm.dtype
        )
        onnx_graph.layer(
            op="Unsqueeze",
            name="UnsqueezeSelectedMaskCoeff",
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
        onnx_graph.layer(
            op="Reshape",
            name="ReshapePooledProto",
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
        onnx_graph.layer(
            op="MatMul",
            name="MatMulMask",
            inputs=[selected_mc_unsq, pooled_proto_flat],
            outputs=[mask_logits],
        )

        mask_logits_sq = gs.Variable(
            name="det_mask_logits_sq", dtype=mask_protos.dtype
        )
        onnx_graph.layer(
            op="Squeeze",
            name="SqueezeMaskLogits",
            inputs=[mask_logits, make_const_int64("mask_logits_axis", [1])],
            outputs=[mask_logits_sq],
        )

        mask_sigmoid = gs.Variable(
            name="det_masks_sigmoid", dtype=mask_protos.dtype
        )
        onnx_graph.layer(
            op="Sigmoid",
            name="SigmoidDetMasks",
            inputs=[mask_logits_sq],
            outputs=[mask_sigmoid],
        )

        mask_shape = gs.Variable(
            name="det_masks_shape", dtype=TensorProto.INT64
        )
        onnx_graph.layer(
            op="Concat",
            name="ConcatDetMaskShape",
            inputs=[
                batch_dim,
                make_const_int64("det_masks_max_dets", [max_dets]),
                make_const_int64("det_masks_h", [mask_h]),
                make_const_int64("det_masks_w", [mask_w]),
            ],
            outputs=[mask_shape],
            attrs=OrderedDict(axis=0),
        )

        det_masks = gs.Variable(
            name="det_masks",
            dtype=mask_protos.dtype,
            shape=[
                "batch" if is_dynamic else batch_size,
                max_dets,
                mask_h,
                mask_w,
            ],
        )
        onnx_graph.layer(
            op="Reshape",
            name="ReshapeDetMasks",
            inputs=[mask_sigmoid, mask_shape],
            outputs=[det_masks],
        )

        onnx_graph.outputs = [
            op_outputs[0],
            op_outputs[1],
            op_outputs[2],
            op_outputs[3],
            det_masks,
        ]
        return

    if kpts_input is None:
        raise ValueError("Pose NMS requires keypoints output")

    kpts_anchor_first = gs.Variable(
        name="kpts_anchor_first", dtype=kpts_input.dtype
    )
    onnx_graph.layer(
        op="Transpose",
        name="TransposeKptsAnchorFirst",
        inputs=[kpts_input],
        outputs=[kpts_anchor_first],
        attrs=OrderedDict(perm=[0, 3, 1, 2]),
    )

    det_kpts_shape = None
    if kpts_input.shape is not None and len(kpts_input.shape) >= 4:
        det_kpts_shape = [
            "batch" if is_dynamic else batch_size,
            max_dets,
            kpts_input.shape[1],
            kpts_input.shape[2],
        ]

    det_kpts = gs.Variable(
        name="det_kpts", dtype=kpts_input.dtype, shape=det_kpts_shape
    )
    onnx_graph.layer(
        op="GatherND",
        name="GatherDetKpts",
        inputs=[kpts_anchor_first, indices_expanded],
        outputs=[det_kpts],
        attrs={"batch_dims": 1},
    )

    onnx_graph.outputs = [
        op_outputs[0],
        op_outputs[1],
        op_outputs[2],
        op_outputs[3],
        det_kpts,
    ]
