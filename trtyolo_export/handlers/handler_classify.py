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
"""Handler for classification-style ONNX export graphs."""

from __future__ import annotations

from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import onnx_graphsurgeon as gs
from onnx import TensorProto

from ..types import GraphContext, MatchResult, RewriteResult
from ..utils import cast_to_dtype, make_const_int64
from .handler_base import BaseHandler


class ClassifyHandler(BaseHandler):
    """Rewrite single-output classification graphs into TopK outputs."""

    name = "classify"

    @staticmethod
    def _match_output_producer(
        ctx: GraphContext,
    ) -> Optional[Tuple[gs.Variable, gs.Node]]:
        if len(ctx.graph.outputs) != 1:
            return None

        output = ctx.graph.outputs[0]
        producer = output.inputs[0] if output.inputs else None
        if producer is None:
            return None

        if producer.op == "Softmax":
            return output, producer

        if producer.op != "Gemm":
            return None

        has_reshape_input = any(
            inp.inputs and inp.inputs[0].op == "Reshape"
            for inp in producer.inputs
            if isinstance(inp, gs.Variable)
        )
        if not has_reshape_input:
            return None

        return output, producer

    @staticmethod
    def _get_classify_scores(ctx: GraphContext) -> gs.Variable:
        match = ClassifyHandler._match_output_producer(ctx)
        if match is None:
            raise ValueError(
                "Classify output is neither Softmax nor Gemm-with-Reshape"
            )

        output, producer = match
        if producer.op == "Softmax":
            return output

        softmax_out = gs.Variable(
            name="classify_softmax", dtype=output.dtype, shape=output.shape
        )
        ctx.graph.layer(
            op="Softmax",
            name="SoftmaxClassify",
            inputs=[output],
            outputs=[softmax_out],
            attrs={"axis": 1},
        )
        return softmax_out

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Match classify graphs with a Softmax or Gemm output head."""
        match = self._match_output_producer(ctx)
        if match is None:
            return None
        _output, producer = match
        if producer.op == "Softmax":
            return MatchResult(
                score=95, reason="Single output Softmax for classify"
            )
        return MatchResult(score=94, reason="Single output Gemm for classify")

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Convert classify outputs to scores and class indices."""
        if len(ctx.graph.outputs) != 1:
            raise ValueError("Classify expects single output")

        classify_scores = self._get_classify_scores(ctx)

        output_shape = classify_scores.shape
        if (
            output_shape is not None
            and len(output_shape) > 1
            and isinstance(output_shape[1], int)
        ):
            k_tensor = make_const_int64(
                "topk_k_const", [min(5, output_shape[1])]
            )
        else:
            cls_shape = gs.Variable(name="cls_shape", dtype=TensorProto.INT64)
            ctx.graph.layer(
                op="Shape",
                name="ShapeClassify",
                inputs=[classify_scores],
                outputs=[cls_shape],
            )

            cls_dim = gs.Variable(name="cls_dim", dtype=TensorProto.INT64)
            ctx.graph.layer(
                op="Slice",
                name="SliceClassifyDim",
                inputs=[
                    cls_shape,
                    make_const_int64("cls_dim_start", [1]),
                    make_const_int64("cls_dim_end", [2]),
                    make_const_int64("cls_dim_axes", [0]),
                ],
                outputs=[cls_dim],
            )

            cls_k_const = make_const_int64("topk_k_const", [5])
            k_tensor = gs.Variable(name="topk_k", dtype=TensorProto.INT64)
            ctx.graph.layer(
                op="Min",
                name="MinTopK",
                inputs=[cls_dim, cls_k_const],
                outputs=[k_tensor],
            )

        topk_values = gs.Variable(
            name="topk_values", dtype=classify_scores.dtype
        )
        topk_indices = gs.Variable(name="topk_indices", dtype=TensorProto.INT64)
        ctx.graph.layer(
            op="TopK",
            name="TopKClassify",
            inputs=[classify_scores, k_tensor],
            outputs=[topk_values, topk_indices],
            attrs={"axis": 1, "largest": 1, "sorted": 1},
        )

        cast_dtype = TensorProto.FLOAT
        if classify_scores.dtype in (np.float16, np.dtype("float16")):
            cast_dtype = TensorProto.FLOAT16
        elif classify_scores.dtype in (np.float32, np.dtype("float32")):
            cast_dtype = TensorProto.FLOAT

        topk_indices_cast = cast_to_dtype(
            ctx.graph,
            topk_indices,
            to_dtype=cast_dtype,
            name="CastTopKIndices",
            out_name="topk_indices_cast",
        )

        topk_unsqueeze_axis = make_const_int64("topk_unsq_axis", [-1])

        values_unsqueeze = gs.Variable(
            name="topk_values_unsqueeze", dtype=classify_scores.dtype
        )
        ctx.graph.layer(
            op="Unsqueeze",
            name="UnsqueezeTopKValues",
            inputs=[topk_values, topk_unsqueeze_axis],
            outputs=[values_unsqueeze],
        )

        indices_unsqueeze = gs.Variable(
            name="topk_indices_unsqueeze", dtype=classify_scores.dtype
        )
        ctx.graph.layer(
            op="Unsqueeze",
            name="UnsqueezeTopKIndices",
            inputs=[topk_indices_cast, topk_unsqueeze_axis],
            outputs=[indices_unsqueeze],
        )

        stacked = gs.Variable(name="topk", dtype=classify_scores.dtype)
        ctx.graph.layer(
            op="Concat",
            name="ConcatTopK",
            inputs=[values_unsqueeze, indices_unsqueeze],
            outputs=[stacked],
            attrs=OrderedDict(axis=-1),
        )

        ctx.graph.outputs = [stacked]
        return RewriteResult(handler_name=self.name, plugin_op=None)
