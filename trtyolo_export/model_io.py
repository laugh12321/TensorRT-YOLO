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
"""ONNX model loading, inference, and serialization helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import onnx
import onnx_graphsurgeon as gs

from .log import logger
from .types import ModelInfo


class ModelIONodeCountError(ValueError):
    """Raised when the model input/output node counts are unsupported."""


def get_default_opset(onnx_model: onnx.ModelProto) -> Optional[int]:
    """Return the default-domain opset version if present."""
    for opset_import in onnx_model.opset_import:
        if opset_import.domain in ("", "ai.onnx"):
            return opset_import.version
    return None


def convert_onnx_opset(
    onnx_model: onnx.ModelProto, target_opset: int
) -> onnx.ModelProto:
    """Convert the model to a target default-domain opset version."""
    current_opset = get_default_opset(onnx_model)
    if current_opset is None:
        raise ValueError(
            "ONNX model does not declare a default-domain opset version."
        )
    if current_opset == target_opset:
        return onnx_model

    logger.info(
        f"Converting ONNX opset from {current_opset} to {target_opset}."
    )
    try:
        return onnx.version_converter.convert_version(onnx_model, target_opset)
    except Exception as e:
        raise ValueError(
            "Failed to convert ONNX opset from "
            f"{current_opset} to {target_opset}: {e}"
        ) from e


def load_onnx_graph(onnx_path: str) -> Tuple[gs.Graph, ModelInfo]:
    """Load an ONNX graph and return (graph, model_info)."""
    onnx_graph = gs.import_onnx(onnx.load(onnx_path))
    assert onnx_graph is not None, "Failed to load ONNX graph!"

    input_count = len(onnx_graph.inputs)
    output_count = len(onnx_graph.outputs)

    input_shape = onnx_graph.inputs[0].shape if input_count == 1 else None
    is_dynamic: Optional[bool] = None
    batch_size: Optional[int] = None
    if input_shape is not None and len(input_shape) == 4:
        if isinstance(input_shape[0], str) or input_shape[0] is None:
            is_dynamic = True
            batch_size = -1
        else:
            is_dynamic = False
            batch_size = int(input_shape[0])

    model_info_parts = [f"Inputs: {input_count}", f"Outputs: {output_count}"]
    if is_dynamic is None:
        model_info_parts.insert(0, "Dynamic batch: Unknown")
    else:
        model_info_parts.insert(0, f"Dynamic batch: {is_dynamic}")
        if not is_dynamic and batch_size is not None:
            model_info_parts.insert(1, f"Batch size: {batch_size}")
    logger.info("Model info - " + ", ".join(model_info_parts))

    io_errors = []
    if input_count != 1:
        io_errors.append(f"expected exactly 1 input node, got {input_count}")
    if output_count not in (1, 2):
        io_errors.append(
            "expected 1 output node, or 2 output nodes for Segment models, "
            f"got {output_count}"
        )
    if io_errors:
        message = "Invalid model I/O: " + "; ".join(io_errors)
        logger.error(message)
        raise ModelIONodeCountError(message)

    assert input_shape is not None and len(input_shape) == 4, (
        "ONNX graph input must be a 4D tensor!"
    )
    assert is_dynamic is not None and batch_size is not None

    onnx_graph.cleanup().toposort()
    try:
        onnx_graph.fold_constants()
    except Exception as e:
        logger.warning(f"Constant folding failed: {e}")

    return onnx_graph, ModelInfo(is_dynamic=is_dynamic, batch_size=batch_size)


def run_shape_inference(onnx_graph: gs.Graph, epochs: int = 2) -> gs.Graph:
    """Run shape inference + constant folding for a few passes."""
    for _ in range(epochs):
        nodes_before = len(onnx_graph.nodes)

        onnx_graph.cleanup().toposort()
        try:
            for node in onnx_graph.nodes:
                for o in node.outputs:
                    if o in onnx_graph.outputs:
                        continue
                    # Let ONNX shape inference recompute unknowns.
                    o.shape = None
            onnx_model = gs.export_onnx(onnx_graph)
            onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
            onnx_graph = gs.import_onnx(onnx_model)
        except Exception as e:
            logger.warning(
                f"Shape inference could not be performed at this time:\n{e}"
            )

        try:
            onnx_graph.fold_constants()
        except Exception as e:
            logger.warning(f"Constant folding failed: {e}")

        nodes_after = len(onnx_graph.nodes)
        if nodes_before == nodes_after:
            break

    return onnx_graph


def normalize_dynamic_batch_output_shapes(
    onnx_graph: gs.Graph, is_dynamic: bool
) -> None:
    """Normalize dynamic output batch dim names to 'batch'."""
    if not is_dynamic:
        return

    for output in onnx_graph.outputs:
        if output.shape is None or len(output.shape) == 0:
            continue
        normalized_shape = list(output.shape)
        normalized_shape[0] = "batch"
        output.shape = normalized_shape


def normalize_dynamic_batch_output_dims(
    onnx_model: onnx.ModelProto, is_dynamic: bool
) -> onnx.ModelProto:
    """Normalize exported dynamic output batch dim params to ``batch``."""
    if not is_dynamic:
        return onnx_model

    for output in onnx_model.graph.output:
        tensor_type = output.type.tensor_type
        if not tensor_type.HasField("shape") or len(tensor_type.shape.dim) == 0:
            continue
        batch_dim = tensor_type.shape.dim[0]
        batch_dim.ClearField("dim_value")
        batch_dim.dim_param = "batch"

    return onnx_model


def save_onnx_graph(
    onnx_graph: gs.Graph,
    output_path: str,
    opset: Optional[int] = None,
    simplify: bool = False,
    is_dynamic: bool = False,
) -> None:
    """Export the graph to ONNX (optionally slim)."""
    onnx_graph.cleanup().toposort()
    normalize_dynamic_batch_output_shapes(onnx_graph, is_dynamic)
    onnx_model = gs.export_onnx(onnx_graph)

    if opset is not None:
        onnx_model = convert_onnx_opset(onnx_model, opset)

    if simplify:
        try:
            import onnxslim

            onnx_model = onnxslim.slim(onnx_model)
            logger.info("Simplified ONNX model.")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")

    onnx_model = normalize_dynamic_batch_output_dims(onnx_model, is_dynamic)
    onnx.save(onnx_model, output_path)
    logger.success(f"Saved converted ONNX to {output_path}")
