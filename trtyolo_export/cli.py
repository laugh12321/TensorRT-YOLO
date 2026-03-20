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
"""Command-line interface for TensorRT-YOLO ONNX conversion."""

from __future__ import annotations

from os.path import normcase
from pathlib import Path
from typing import Optional

import rich_click as click

from . import __version__


def validate_onnx_output(
    ctx: click.Context, param: click.Parameter, value: str
) -> str:
    """Validate that the output path uses an ONNX file extension."""
    del ctx, param
    output = Path(value)
    if output.suffix.lower() != ".onnx":
        raise click.BadParameter("Output file must use the .onnx suffix.")
    return str(output)


def normalize_output_path(input_path: str, output_path: str) -> Path:
    """Prevent the converted model from overwriting the source ONNX file."""
    from .log import logger

    input_file = Path(input_path)
    output_file = Path(output_path)

    input_key = normcase(str(input_file.resolve()))
    output_key = normcase(str(output_file.resolve()))
    if input_key != output_key:
        return output_file

    adjusted_output = output_file.with_name(
        f"{output_file.stem}-trtyolo{output_file.suffix}"
    )
    logger.warning(
        "Output path matches input path. Added '-trtyolo' suffix "
        f"to avoid overwriting the source file: {adjusted_output.name}"
    )
    return adjusted_output


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Convert an exported ONNX graph into TensorRT-YOLO compatible outputs. "
        "The graph type is detected automatically. "
        "Example: trtyolo-export -i model.onnx -o model-trtyolo.onnx"
    ),
)
@click.version_option(version=__version__, prog_name="trtyolo-export")
@click.option(
    "--verbose/--quiet", default=True, help="Show conversion progress logs."
)
@click.option(
    "-i",
    "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
    required=True,
    help="Path to the source ONNX file.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(file_okay=True, dir_okay=False, path_type=str),
    required=True,
    callback=validate_onnx_output,
    help=(
        "Path to save the converted ONNX file. If it matches the input "
        "path, '-trtyolo' is appended automatically."
    ),
)
@click.option(
    "--opset",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "Target ONNX opset version. Defaults to preserving the source "
        "model opset."
    ),
)
@click.option(
    "--max-dets",
    "--max_dets",
    default=100,
    type=click.IntRange(min=1),
    show_default=True,
    help="Maximum detections used when appending TensorRT NMS plugins.",
)
@click.option(
    "--conf-thres",
    "--conf_thres",
    default=0.25,
    type=float,
    show_default=True,
    help=(
        "Confidence threshold used by plugin-based and NMS-free "
        "postprocess outputs."
    ),
)
@click.option(
    "--iou-thres",
    "--iou_thres",
    default=0.45,
    type=float,
    show_default=True,
    help="IoU threshold used when appending TensorRT NMS plugins.",
)
@click.option(
    "-s",
    "--simplify",
    is_flag=True,
    help="Run onnxslim after conversion.",
)
def main(
    verbose: bool,
    input_path: str,
    output: str,
    opset: Optional[int],
    max_dets: int,
    conf_thres: float,
    iou_thres: float,
    simplify: bool,
) -> None:
    """CLI for the ONNX conversion pipeline used by TensorRT-YOLO."""
    from .log import configure_logging, logger
    from .model_io import ModelIONodeCountError
    from .pipeline import RewriteConfig, UnsupportedModelError, rewrite_onnx

    configure_logging(verbose)
    output_path = normalize_output_path(input_path, output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config = RewriteConfig(
        opset=opset,
        max_dets=max_dets,
        score_thresh=conf_thres,
        iou_thresh=iou_thres,
        simplify=simplify,
        verbose=verbose,
    )

    try:
        rewrite_onnx(
            onnx_path=input_path,
            output_path=str(output_path),
            config=config,
        )
    except (ModelIONodeCountError, UnsupportedModelError) as e:
        logger.error(str(e))
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
