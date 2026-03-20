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
"""Utility helpers for ONNX graph inspection and tensor rewrites."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Sequence

import numpy as np
import onnx_graphsurgeon as gs


def find_final_concat_node(onnx_graph: gs.Graph) -> gs.Node:
    """Return the final head node used as the model output."""
    if len(onnx_graph.outputs) == 0:
        raise AssertionError("Output count mismatch!")

    last_node = onnx_graph.nodes[-1]
    if last_node.op in {"Concat", "Gemm"}:
        return last_node

    for output in onnx_graph.outputs:
        producer = output.inputs[0] if output.inputs else None
        if producer is not None and producer.op == "Concat":
            return producer

    for output in onnx_graph.outputs:
        producer = output.inputs[0] if output.inputs else None
        if producer is not None and producer.op == "Softmax":
            return producer

    for output in onnx_graph.outputs:
        producer = output.inputs[0] if output.inputs else None
        if producer is not None and producer.op == "Gemm":
            return producer

    for node in reversed(onnx_graph.nodes):
        if node.op == "Concat":
            return node

    raise AssertionError("Last node is not a Concat, Softmax, or Gemm!")


def make_const_int64(name: str, values: Sequence[int]) -> gs.Constant:
    """Create an int64 constant tensor."""
    return gs.Constant(name=name, values=np.array(values, dtype=np.int64))


def make_const_float32(name: str, value: float) -> gs.Constant:
    """Create a float32 scalar constant tensor."""
    return gs.Constant(name=name, values=np.array([value], dtype=np.float32))


def slice_axis_range(
    onnx_graph: gs.Graph,
    input_tensor: gs.Variable,
    start: int,
    end: int,
    axis: int,
    name: str,
    out_name: str,
) -> gs.Variable:
    """Slice a tensor on a single axis [start, end)."""
    out = gs.Variable(name=out_name, dtype=input_tensor.dtype)
    onnx_graph.layer(
        op="Slice",
        name=name,
        inputs=[
            input_tensor,
            make_const_int64(f"{name}_start", [start]),
            make_const_int64(f"{name}_end", [end]),
            make_const_int64(f"{name}_axes", [axis]),
        ],
        outputs=[out],
    )
    return out


def squeeze_axis(
    onnx_graph: gs.Graph,
    input_tensor: gs.Variable,
    axis: int,
    name: str,
    out_name: str,
    dtype: Optional[int] = None,
) -> gs.Variable:
    """Squeeze a single axis."""
    out = gs.Variable(name=out_name, dtype=dtype or input_tensor.dtype)
    onnx_graph.layer(
        op="Squeeze",
        name=name,
        inputs=[input_tensor, make_const_int64(f"{name}_axis", [axis])],
        outputs=[out],
    )
    return out


def cast_to_dtype(
    onnx_graph: gs.Graph,
    input_tensor: gs.Variable,
    to_dtype: int,
    name: str,
    out_name: str,
) -> gs.Variable:
    """Cast a tensor to the given ONNX dtype."""
    out = gs.Variable(name=out_name, dtype=to_dtype)
    onnx_graph.layer(
        op="Cast",
        name=name,
        inputs=[input_tensor],
        outputs=[out],
        attrs={"to": to_dtype},
    )
    return out


def transpose_bnc_to_bcn(
    onnx_graph: gs.Graph, input_tensor: gs.Variable
) -> gs.Variable:
    """Transpose [B, N, C] -> [B, C, N] when shape is available."""
    in_shape = input_tensor.shape
    out_shape = None
    if in_shape is not None and len(in_shape) >= 3:
        out_shape = [in_shape[0], in_shape[2], in_shape[1]]

    out = gs.Variable(
        name=f"{input_tensor.name}/transpose_output",
        dtype=input_tensor.dtype,
        shape=out_shape,
    )
    onnx_graph.layer(
        op="Transpose",
        name=f"{input_tensor.name}/transpose",
        inputs=[input_tensor],
        outputs=[out],
        attrs=OrderedDict(perm=[0, 2, 1]),
    )
    return out


def get_static_dim(shape: Optional[Sequence], axis: int) -> Optional[int]:
    """Read a static integer dimension from shape[axis], if available."""
    if shape is None:
        return None

    if axis < 0:
        axis += len(shape)
    if axis < 0 or len(shape) <= axis:
        return None

    dim = shape[axis]
    if isinstance(dim, np.integer):
        return int(dim)
    if isinstance(dim, int):
        return dim
    return None


def get_split_sizes_from_last_split(onnx_graph: gs.Graph) -> Sequence[int]:
    """Get split sizes from the last Split node."""
    split_node = next(
        (node for node in reversed(onnx_graph.nodes) if node.op == "Split"),
        None,
    )
    if split_node is None:
        raise ValueError("Split node not found")

    split_dim = None
    if len(split_node.inputs) == 2:
        split_dim = next(
            (
                inp.values
                for inp in split_node.inputs
                if hasattr(inp, "values") and inp.values is not None
            ),
            None,
        )
    if split_dim is None:
        split_dim = split_node.attrs.get("split")

    if split_dim is None:
        raise ValueError(
            f"Invalid Split node '{split_node.name}': "
            "No valid split dimension found"
        )

    if isinstance(split_dim, np.ndarray):
        split_dim = split_dim.tolist()
    return split_dim


def infer_num_classes_from_concat(
    onnx_graph: gs.Graph, concat_shape: Sequence
) -> int:
    """Infer class count from concat output shape or Split op."""
    if concat_shape is None or len(concat_shape) < 3:
        raise ValueError("Concat output shape is not available")

    _, anchors, features = concat_shape
    if isinstance(anchors, str):
        split_dim = get_split_sizes_from_last_split(onnx_graph)
        return int(split_dim[-1]) - 1

    return int(features) - 5


def convert_xywh_to_xyxy(
    onnx_graph: gs.Graph,
    boxes_xywh: gs.Variable,
    axis: int,
    name_prefix: str,
) -> gs.Variable:
    """Convert boxes from xywh to xyxy along the given feature axis."""
    xy = slice_axis_range(
        onnx_graph,
        boxes_xywh,
        start=0,
        end=2,
        axis=axis,
        name=f"{name_prefix}_slice_xy",
        out_name=f"{name_prefix}_xy",
    )
    wh = slice_axis_range(
        onnx_graph,
        boxes_xywh,
        start=2,
        end=4,
        axis=axis,
        name=f"{name_prefix}_slice_wh",
        out_name=f"{name_prefix}_wh",
    )

    half_wh = gs.Variable(
        name=f"{name_prefix}_half_wh",
        dtype=boxes_xywh.dtype,
    )
    onnx_graph.layer(
        op="Mul",
        name=f"{name_prefix}_mul_half",
        inputs=[wh, make_const_float32(f"{name_prefix}_half_const", 0.5)],
        outputs=[half_wh],
    )

    xy_min = gs.Variable(
        name=f"{name_prefix}_xy_min",
        dtype=boxes_xywh.dtype,
    )
    onnx_graph.layer(
        op="Sub",
        name=f"{name_prefix}_sub_xy_min",
        inputs=[xy, half_wh],
        outputs=[xy_min],
    )

    xy_max = gs.Variable(
        name=f"{name_prefix}_xy_max",
        dtype=boxes_xywh.dtype,
    )
    onnx_graph.layer(
        op="Add",
        name=f"{name_prefix}_add_xy_max",
        inputs=[xy, half_wh],
        outputs=[xy_max],
    )

    boxes_xyxy = gs.Variable(
        name=f"{name_prefix}_xyxy",
        dtype=boxes_xywh.dtype,
    )
    onnx_graph.layer(
        op="Concat",
        name=f"{name_prefix}_concat_xyxy",
        inputs=[xy_min, xy_max],
        outputs=[boxes_xyxy],
        attrs=OrderedDict(axis=axis),
    )
    return boxes_xyxy
