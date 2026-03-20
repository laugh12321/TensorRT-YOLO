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
"""High-level ONNX rewrite pipeline selection and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import onnx_graphsurgeon as gs

from .handlers import BaseHandler, default_handlers
from .log import logger
from .model_io import load_onnx_graph, run_shape_inference, save_onnx_graph
from .plugins import append_trt_nms_plugin
from .types import GraphContext, MatchResult, RewriteResult
from .utils import find_final_concat_node


class UnsupportedModelError(ValueError):
    """Raised when the ONNX graph does not match a supported model family."""


@dataclass(frozen=True)
class RewriteConfig:
    """Runtime options controlling graph rewriting and serialization."""

    opset: Optional[int] = None
    max_dets: int = 100
    score_thresh: float = 0.25
    iou_thresh: float = 0.45
    simplify: bool = False
    infer_epochs: int = 2
    verbose: bool = True


def _describe_handler(handler: BaseHandler) -> str:
    """Return a user-friendly model type description for logs."""
    name = handler.name
    if name == "classify":
        return "Classify Model"

    if "pose" in name:
        base = "Pose Model"
    elif "segment" in name:
        base = "Segment Model"
    elif "obb" in name:
        base = "OBB Model"
    else:
        base = "Detect Model"

    if name.startswith("nms_free_"):
        return f"NMS-free {base}"
    return base


def _build_context(
    graph: gs.Graph, model_info, score_thresh: float
) -> GraphContext:
    final_concat = find_final_concat_node(graph)

    concat_output = None
    mask_protos = None
    if len(graph.outputs) == 1:
        concat_output = (
            final_concat.outputs[0]
            if getattr(final_concat, "outputs", None)
            else graph.outputs[0]
        )
    else:
        for output in graph.outputs:
            producer = output.inputs[0] if output.inputs else None
            if producer is final_concat:
                concat_output = output
            else:
                mask_protos = output

    if concat_output is None:
        concat_output = (
            final_concat.outputs[0]
            if getattr(final_concat, "outputs", None)
            else graph.outputs[0]
        )

    return GraphContext(
        graph=graph,
        model_info=model_info,
        score_thresh=score_thresh,
        final_concat=final_concat,
        concat_output=concat_output,
        mask_protos=mask_protos,
    )


def _collect_matches(
    ctx: GraphContext, handlers: Sequence[BaseHandler]
) -> List[Tuple[BaseHandler, MatchResult]]:
    matches: List[Tuple[BaseHandler, MatchResult]] = []
    for handler in handlers:
        try:
            match = handler.match(ctx)
        except Exception as e:
            logger.debug(f"Skip handler '{handler.name}' during match: {e}")
            continue
        if match is not None:
            matches.append((handler, match))

    matches.sort(key=lambda item: item[1].score, reverse=True)
    return matches


def _segment_is_dynamic(ctx: GraphContext) -> bool:
    return ctx.model_info.is_dynamic


def _select_handlers_for_graph(
    ctx: GraphContext, handlers: Sequence[BaseHandler]
) -> List[Tuple[BaseHandler, MatchResult]]:
    output_count = len(ctx.graph.outputs)

    if output_count == 2:
        segment_handlers = [
            handler for handler in handlers if "segment" in handler.name
        ]
        matches = _collect_matches(ctx, segment_handlers)
        if not matches:
            raise UnsupportedModelError(
                "Two-output models must match a supported Segment pattern."
            )
        if _segment_is_dynamic(ctx):
            logger.warning(
                "Segment models do not support dynamic batch because "
                "ROIAlign requires fixed values."
            )
            raise UnsupportedModelError(
                "Only static-batch Segment models are supported."
            )
        return matches

    output = ctx.graph.outputs[0]
    producer = output.inputs[0] if output.inputs else None
    if producer is None:
        raise UnsupportedModelError(
            "Single-output model is missing an output producer."
        )

    if producer.op == "Concat":
        concat_handlers = [
            handler
            for handler in handlers
            if handler.name != "classify" and "segment" not in handler.name
        ]
        matches = _collect_matches(ctx, concat_handlers)
        if not matches:
            raise UnsupportedModelError(
                "Concat-output models must match a supported Detect, OBB, "
                "or Pose pattern."
            )
        return matches

    if producer.op in {"Softmax", "Gemm"}:
        classify_handlers = [
            handler for handler in handlers if handler.name == "classify"
        ]
        matches = _collect_matches(ctx, classify_handlers)
        if not matches:
            raise UnsupportedModelError(
                f"{producer.op}-output models must match a supported "
                "Classify pattern."
            )
        return matches

    raise UnsupportedModelError(
        f"Unsupported output producer '{producer.op}'. "
        "Expected Concat, Softmax, or Gemm."
    )


def rewrite_onnx(
    onnx_path: str,
    output_path: str,
    config: Optional[RewriteConfig] = None,
    handlers: Optional[Sequence[BaseHandler]] = None,
) -> RewriteResult:
    """Rewrite an exported ONNX graph into TensorRT-YOLO-compatible outputs."""
    config = config or RewriteConfig()
    handlers = handlers or default_handlers()

    base_graph, model_info = load_onnx_graph(onnx_path)
    base_ctx = _build_context(base_graph, model_info, config.score_thresh)

    matches = _select_handlers_for_graph(base_ctx, handlers)

    if config.verbose:
        summary = ", ".join(
            dict.fromkeys(
                _describe_handler(handler) for handler, _match in matches
            )
        )
        logger.info(f"Match candidates: {summary}")

    errors: List[str] = []
    for idx, (handler, _match) in enumerate(matches):
        if idx == 0:
            graph, info = base_graph, model_info
        else:
            graph, info = load_onnx_graph(onnx_path)
        ctx = _build_context(graph, info, config.score_thresh)

        try:
            result = handler.rewrite(ctx)
            has_trt_ops = any(node.op.endswith("_TRT") for node in graph.nodes)
            if config.infer_epochs > 0 and not has_trt_ops:
                graph = run_shape_inference(graph, epochs=config.infer_epochs)

            if result.plugin_op is not None:
                append_trt_nms_plugin(
                    graph,
                    info.is_dynamic,
                    info.batch_size,
                    max_dets=config.max_dets,
                    score_thresh=config.score_thresh,
                    iou_thresh=config.iou_thresh,
                    plugin_op=result.plugin_op,
                )

            save_onnx_graph(
                graph,
                output_path,
                opset=config.opset,
                simplify=config.simplify,
                is_dynamic=info.is_dynamic,
            )
            return result
        except Exception as e:
            errors.append(f"{_describe_handler(handler)}: {e}")
            if config.verbose:
                logger.warning(
                    f"{_describe_handler(handler)} conversion attempt "
                    f"failed: {e}"
                )

    raise ValueError("All matched handlers failed: " + " | ".join(errors))
