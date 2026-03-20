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
"""Shared matching and graph-building helpers used by multiple handlers."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import onnx_graphsurgeon as gs
from onnx import TensorProto

from ..types import GraphContext
from ..utils import (
    cast_to_dtype,
    convert_xywh_to_xyxy,
    get_split_sizes_from_last_split,
    get_static_dim,
    make_const_float32,
    make_const_int64,
    squeeze_axis,
)


def match_legacy_obb_head_inputs(
    concat_node: gs.Node,
) -> Optional[Tuple[gs.Variable, gs.Variable, gs.Variable]]:
    """Match legacy OBB head: dbox Mul, angle Mul, and score Sigmoid."""
    if concat_node.op != "Concat":
        return None
    if len(concat_node.inputs) != 3:
        return None

    mul_tensors = []
    sigmoid_tensors = []
    for idx, input_tensor in enumerate(concat_node.inputs):
        producer = concat_node.i(idx)
        if producer is None:
            return None
        if producer.op == "Mul":
            mul_tensors.append(input_tensor)
        elif producer.op == "Sigmoid":
            sigmoid_tensors.append(input_tensor)
        else:
            return None

    if len(mul_tensors) != 2 or len(sigmoid_tensors) != 1:
        return None

    mul_dims = [get_static_dim(t.shape, 1) for t in mul_tensors]
    if sorted(mul_dims) != [1, 4]:
        return None

    angle_tensor = mul_tensors[mul_dims.index(1)]
    dbox_tensor = mul_tensors[mul_dims.index(4)]
    return dbox_tensor, angle_tensor, sigmoid_tensors[0]


def match_nms_free_obb_postprocess_inputs(
    concat_node: gs.Node,
) -> Optional[
    Tuple[
        gs.Variable,
        gs.Variable,
        gs.Variable,
        gs.Variable,
        Optional[gs.Node],
        int,
    ]
]:
    """Match YOLO26-style OBB postprocess Concat."""
    if len(concat_node.inputs) != 4:
        return None

    concat_axis = int(concat_node.attrs.get("axis", -1))
    concat_dim = get_static_dim(concat_node.outputs[0].shape, concat_axis)
    if concat_dim != 7:
        return None

    gather_tensors = []
    score_out = None
    class_out = None
    class_cast_node = None

    for idx, input_tensor in enumerate(concat_node.inputs):
        producer = concat_node.i(idx)
        name = input_tensor.name or ""

        is_gather = (
            producer is not None and producer.op == "GatherElements"
        ) or ("/GatherElements" in name)
        is_unsqueeze = (
            producer is not None and producer.op == "Unsqueeze"
        ) or ("/Unsqueeze" in name)
        is_cast = (
            producer is not None and producer.op == "Cast"
        ) or "/Cast" in name
        if int(is_gather) + int(is_unsqueeze) + int(is_cast) != 1:
            return None

        if is_gather:
            gather_tensors.append(input_tensor)
        elif is_unsqueeze:
            score_out = input_tensor
        else:
            class_out = input_tensor
            class_cast_node = producer

    if len(gather_tensors) != 2 or score_out is None or class_out is None:
        return None

    gather_dims = [get_static_dim(t.shape, concat_axis) for t in gather_tensors]
    if gather_dims.count(4) != 1 or gather_dims.count(1) != 1:
        return None

    dbox_out = gather_tensors[gather_dims.index(4)]
    angle_out = gather_tensors[gather_dims.index(1)]
    return (
        dbox_out,
        angle_out,
        score_out,
        class_out,
        class_cast_node,
        concat_axis,
    )


def resolve_nms_free_detect_inputs(
    concat_node: gs.Node,
) -> Tuple[gs.Variable, gs.Variable, gs.Variable]:
    """Resolve detect outputs from an NMS-free postprocess head."""
    if len(concat_node.inputs) != 3:
        raise ValueError(
            "NMS-free detect Concat must have 3 inputs, "
            f"got {len(concat_node.inputs)}"
        )

    boxes_out = None
    score_out = None
    class_out = None
    class_cast_node = None

    for idx, input_tensor in enumerate(concat_node.inputs):
        producer = concat_node.i(idx)
        name = input_tensor.name or ""

        if (
            producer is not None and producer.op == "GatherElements"
        ) or "/GatherElements" in name:
            if boxes_out is not None:
                raise ValueError(
                    "NMS-free detect expects exactly one GatherElements input"
                )
            boxes_out = input_tensor
            continue

        if (
            producer is not None and producer.op == "Unsqueeze"
        ) or "/Unsqueeze" in name:
            if score_out is not None:
                raise ValueError(
                    "NMS-free detect expects exactly one Unsqueeze input"
                )
            score_out = input_tensor
            continue

        if (producer is not None and producer.op == "Cast") or "/Cast" in name:
            if class_out is not None:
                raise ValueError(
                    "NMS-free detect expects exactly one Cast input"
                )
            class_out = input_tensor
            class_cast_node = producer

    if boxes_out is None or score_out is None or class_out is None:
        names = [inp.name for inp in concat_node.inputs]
        raise ValueError(
            f"Unable to resolve NMS-free detect inputs from names: {names}"
        )

    if class_cast_node is not None:
        class_cast_node.attrs["to"] = TensorProto.INT32
    class_out.dtype = TensorProto.INT32
    return boxes_out, score_out, class_out


def resolve_nms_free_obb_inputs(
    onnx_graph: gs.Graph,
    concat_node: gs.Node,
) -> Tuple[gs.Variable, gs.Variable, gs.Variable]:
    """Resolve OBB outputs and assemble rotated boxes."""
    matched = match_nms_free_obb_postprocess_inputs(concat_node)
    if matched is None:
        names = [inp.name for inp in concat_node.inputs]
        raise ValueError(
            "Unable to resolve YOLO26 OBB postprocess inputs "
            f"from names: {names}"
        )

    dbox_out, angle_out, score_out, class_out, class_cast_node, concat_axis = (
        matched
    )
    dbox_xyxy = convert_xywh_to_xyxy(
        onnx_graph=onnx_graph,
        boxes_xywh=dbox_out,
        axis=concat_axis,
        name_prefix="PostprocessOBB_dbox",
    )

    det_boxes = gs.Variable(name="det_boxes", dtype=dbox_out.dtype)
    onnx_graph.layer(
        op="Concat",
        name="ConcatPostprocessRBox",
        inputs=[dbox_xyxy, angle_out],
        outputs=[det_boxes],
        attrs=OrderedDict(axis=concat_axis),
    )

    if class_cast_node is not None:
        class_cast_node.attrs["to"] = TensorProto.INT32
    class_out.dtype = TensorProto.INT32
    return det_boxes, score_out, class_out


def match_nms_free_pose_postprocess_inputs(
    concat_node: gs.Node,
) -> Optional[
    Tuple[
        gs.Variable,
        gs.Variable,
        gs.Variable,
        gs.Variable,
        Optional[gs.Node],
        int,
    ]
]:
    """Match NMS-free pose postprocess Concat."""
    if len(concat_node.inputs) != 4:
        return None

    concat_axis = int(concat_node.attrs.get("axis", -1))
    gather_tensors = []
    score_out = None
    class_out = None
    class_cast_node = None

    for idx, input_tensor in enumerate(concat_node.inputs):
        producer = concat_node.i(idx)
        name = input_tensor.name or ""

        is_gather = (
            producer is not None and producer.op == "GatherElements"
        ) or ("/GatherElements" in name)
        is_unsqueeze = (
            producer is not None and producer.op == "Unsqueeze"
        ) or ("/Unsqueeze" in name)
        is_cast = (
            producer is not None and producer.op == "Cast"
        ) or "/Cast" in name
        if int(is_gather) + int(is_unsqueeze) + int(is_cast) != 1:
            return None

        if is_gather:
            gather_tensors.append(input_tensor)
        elif is_unsqueeze:
            score_out = input_tensor
        else:
            class_out = input_tensor
            class_cast_node = producer

    if len(gather_tensors) != 2 or score_out is None or class_out is None:
        return None

    gather_dims = [get_static_dim(t.shape, concat_axis) for t in gather_tensors]
    if gather_dims.count(4) != 1:
        return None

    other_dim = gather_dims[1 - gather_dims.index(4)]
    if other_dim is None or other_dim == 1:
        return None

    dbox_out = gather_tensors[gather_dims.index(4)]
    kpts_out = gather_tensors[1 - gather_dims.index(4)]
    return (
        dbox_out,
        kpts_out,
        score_out,
        class_out,
        class_cast_node,
        concat_axis,
    )


def resolve_nms_free_pose_inputs(
    concat_node: gs.Node,
) -> Tuple[gs.Variable, gs.Variable, gs.Variable, gs.Variable]:
    """Resolve boxes, keypoints, scores, and classes for pose outputs."""
    matched = match_nms_free_pose_postprocess_inputs(concat_node)
    if matched is None:
        names = [inp.name for inp in concat_node.inputs]
        raise ValueError(
            f"Unable to resolve NMS-free pose inputs from names: {names}"
        )

    dbox_out, kpts_out, score_out, class_out, class_cast_node, _ = matched
    if class_cast_node is not None:
        class_cast_node.attrs["to"] = TensorProto.INT32
    class_out.dtype = TensorProto.INT32
    return dbox_out, kpts_out, score_out, class_out


def match_nms_free_segment_postprocess_inputs(
    concat_node: gs.Node,
    mask_dim: Optional[int],
) -> Optional[
    Tuple[
        gs.Variable,
        gs.Variable,
        gs.Variable,
        gs.Variable,
        Optional[gs.Node],
        int,
    ]
]:
    """Match NMS-free segment postprocess Concat."""
    if len(concat_node.inputs) != 4:
        return None

    concat_axis = int(concat_node.attrs.get("axis", -1))
    gather_tensors = []
    score_out = None
    class_out = None
    class_cast_node = None

    for idx, input_tensor in enumerate(concat_node.inputs):
        producer = concat_node.i(idx)
        name = input_tensor.name or ""

        is_gather = (
            producer is not None and producer.op == "GatherElements"
        ) or ("/GatherElements" in name)
        is_unsqueeze = (
            producer is not None and producer.op == "Unsqueeze"
        ) or ("/Unsqueeze" in name)
        is_cast = (
            producer is not None and producer.op == "Cast"
        ) or "/Cast" in name
        if int(is_gather) + int(is_unsqueeze) + int(is_cast) != 1:
            return None

        if is_gather:
            gather_tensors.append(input_tensor)
        elif is_unsqueeze:
            score_out = input_tensor
        else:
            class_out = input_tensor
            class_cast_node = producer

    if len(gather_tensors) != 2 or score_out is None or class_out is None:
        return None

    gather_dims = [get_static_dim(t.shape, concat_axis) for t in gather_tensors]
    if gather_dims.count(4) != 1:
        return None

    other_dim = gather_dims[1 - gather_dims.index(4)]
    if other_dim is None or other_dim == 1:
        return None
    if mask_dim is not None and other_dim != mask_dim:
        return None

    dbox_out = gather_tensors[gather_dims.index(4)]
    mask_coeffs_out = gather_tensors[1 - gather_dims.index(4)]
    return (
        dbox_out,
        mask_coeffs_out,
        score_out,
        class_out,
        class_cast_node,
        concat_axis,
    )


def resolve_nms_free_segment_inputs(
    concat_node: gs.Node,
    mask_dim: Optional[int],
) -> Tuple[gs.Variable, gs.Variable, gs.Variable, gs.Variable]:
    """Resolve boxes, masks, scores, and classes for segment outputs."""
    matched = match_nms_free_segment_postprocess_inputs(concat_node, mask_dim)
    if matched is None:
        names = [inp.name for inp in concat_node.inputs]
        raise ValueError(
            f"Unable to resolve NMS-free segment inputs from names: {names}"
        )

    dbox_out, mask_coeffs_out, score_out, class_out, class_cast_node, _ = (
        matched
    )
    if class_cast_node is not None:
        class_cast_node.attrs["to"] = TensorProto.INT32
    class_out.dtype = TensorProto.INT32
    return dbox_out, mask_coeffs_out, score_out, class_out


def find_penultimate_concat_node(onnx_graph: gs.Graph) -> gs.Node:
    """Return the second-to-last Concat node in the graph."""
    concat_nodes = [
        node for node in reversed(onnx_graph.nodes) if node.op == "Concat"
    ]
    if len(concat_nodes) < 2:
        raise ValueError("Unable to find penultimate Concat node")
    return concat_nodes[1]


def resolve_pose_kpts_shape_source(onnx_graph: gs.Graph) -> gs.Variable:
    """Find the reference tensor used to recover pose keypoint shape."""
    prev_concat = find_penultimate_concat_node(onnx_graph)
    if len(prev_concat.inputs) != 3:
        raise ValueError(
            "Penultimate Concat must have 3 inputs for pose shape resolve"
        )

    reshape_node = None
    for idx in range(len(prev_concat.inputs)):
        producer = prev_concat.i(idx)
        if producer is not None and producer.op == "Reshape":
            reshape_node = producer
            break

    if reshape_node is None or len(reshape_node.inputs) == 0:
        raise ValueError("Reshape node not found in penultimate Concat inputs")
    return reshape_node.inputs[0]


def reshape_pose_kpts(
    onnx_graph: gs.Graph,
    kpts_out: gs.Variable,
    shape_source: gs.Variable,
) -> gs.Variable:
    """Reshape flattened keypoints back to their structured pose layout."""
    shape_kpts = gs.Variable(name="kpts_shape", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Shape", name="ShapeKpts", inputs=[kpts_out], outputs=[shape_kpts]
    )

    batch_dim = gs.Variable(name="kpts_batch_dim", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Slice",
        name="SliceKptsBatch",
        inputs=[
            shape_kpts,
            make_const_int64("kpts_batch_start", [0]),
            make_const_int64("kpts_batch_end", [1]),
            make_const_int64("kpts_batch_axes", [0]),
        ],
        outputs=[batch_dim],
    )

    n_dim = gs.Variable(name="kpts_n_dim", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Slice",
        name="SliceKptsN",
        inputs=[
            shape_kpts,
            make_const_int64("kpts_n_start", [1]),
            make_const_int64("kpts_n_end", [2]),
            make_const_int64("kpts_n_axes", [0]),
        ],
        outputs=[n_dim],
    )

    shape_ref = gs.Variable(name="pose_ref_shape", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Shape",
        name="ShapePoseRef",
        inputs=[shape_source],
        outputs=[shape_ref],
    )

    a_dim = gs.Variable(name="pose_a_dim", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Slice",
        name="SlicePoseA",
        inputs=[
            shape_ref,
            make_const_int64("pose_a_start", [1]),
            make_const_int64("pose_a_end", [2]),
            make_const_int64("pose_a_axes", [0]),
        ],
        outputs=[a_dim],
    )

    b_dim = gs.Variable(name="pose_b_dim", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Slice",
        name="SlicePoseB",
        inputs=[
            shape_ref,
            make_const_int64("pose_b_start", [2]),
            make_const_int64("pose_b_end", [3]),
            make_const_int64("pose_b_axes", [0]),
        ],
        outputs=[b_dim],
    )

    new_shape = gs.Variable(name="kpts_new_shape", dtype=TensorProto.INT64)
    onnx_graph.layer(
        op="Concat",
        name="ConcatKptsShape",
        inputs=[batch_dim, n_dim, a_dim, b_dim],
        outputs=[new_shape],
        attrs=OrderedDict(axis=0),
    )

    det_kpts = gs.Variable(name="det_kpts", dtype=kpts_out.dtype)
    onnx_graph.layer(
        op="Reshape",
        name="ReshapeDetKpts",
        inputs=[kpts_out, new_shape],
        outputs=[det_kpts],
    )
    return det_kpts


def resolve_mask_protos(ctx: GraphContext) -> gs.Variable:
    """Return the mask prototype output for segmentation models."""
    if ctx.mask_protos is None:
        raise ValueError("Mask protos output not found for segmentation model")
    return ctx.mask_protos


def resolve_segment_feature_axis(
    concat_output: gs.Variable, base_dim: int, mask_dim: int
) -> int:
    """Infer which axis stores per-anchor segment features."""
    if concat_output.shape is None or len(concat_output.shape) < 3:
        return 2

    dim2 = get_static_dim(concat_output.shape, 2)
    dim1 = get_static_dim(concat_output.shape, 1)
    threshold = base_dim + mask_dim
    if dim2 is not None and dim2 >= threshold:
        return 2
    if dim1 is not None and dim1 >= threshold:
        return 1
    return 2


def infer_segment_num_classes(
    onnx_graph: gs.Graph,
    concat_output: gs.Variable,
    mask_dim: int,
    base_dim: int,
    feature_axis: int,
) -> int:
    """Infer the number of segmentation classes from graph metadata."""
    features = get_static_dim(concat_output.shape, feature_axis)
    if features is None:
        split_sizes = get_split_sizes_from_last_split(onnx_graph)
        features = int(sum(split_sizes))

    num_classes = features - base_dim - mask_dim
    if num_classes <= 0:
        raise ValueError(
            "Invalid segment class count inferred: "
            f"features={features}, base={base_dim}, mask_dim={mask_dim}"
        )
    return num_classes


def build_nms_free_common_outputs(
    onnx_graph: gs.Graph,
    score_out: gs.Variable,
    class_out: gs.Variable,
    score_thresh: float,
) -> Tuple[gs.Variable, gs.Variable, gs.Variable]:
    """Build num_dets, det_scores and det_classes for NMS-free outputs."""
    det_scores = squeeze_axis(
        onnx_graph,
        score_out,
        axis=2,
        name="squeeze_scores",
        out_name="det_scores",
    )
    det_classes = squeeze_axis(
        onnx_graph,
        class_out,
        axis=2,
        name="squeeze_classes",
        out_name="det_classes",
        dtype=TensorProto.INT32,
    )

    score_mask = gs.Variable(name="score_mask", dtype=TensorProto.BOOL)
    onnx_graph.layer(
        op="GreaterOrEqual",
        name="score_threshold_filter",
        inputs=[
            det_scores,
            make_const_float32("conf_thres_const", score_thresh),
        ],
        outputs=[score_mask],
    )

    mask_int32 = cast_to_dtype(
        onnx_graph,
        score_mask,
        to_dtype=TensorProto.INT32,
        name="cast_mask_to_int32",
        out_name="mask_int32",
    )

    num_dets = gs.Variable(name="num_dets", dtype=TensorProto.INT32)
    onnx_graph.layer(
        op="ReduceSum",
        name="count_valid_detections",
        inputs=[mask_int32, make_const_int64("sum_axis", [1])],
        outputs=[num_dets],
        attrs={"keepdims": 1},
    )
    return num_dets, det_scores, det_classes
