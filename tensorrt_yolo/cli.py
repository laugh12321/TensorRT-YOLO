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
# File    :   cli.py
# Version :   6.2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:26:53
# Desc    :   trtyolo cli.
# ==============================================================================
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import rich_click as click
from loguru import logger

logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])


def validate_imgsz(ctx: click.Context, param: click.Parameter, value: str) -> Tuple[int, int]:
    """Validate and parse the imgsz parameter."""
    try:
        if ',' in value:
            h, w = map(int, value.split(','))
            return (h, w)
        size = int(value)
        return (size, size)
    except ValueError:
        raise click.BadParameter('Image size must be in format "size" or "height,width" (e.g., 640 or 640,480)')


def validate_names(ctx: click.Context, param: click.Parameter, value: str) -> Optional[List[str]]:
    """Validate and parse the names parameter."""
    if value is None:
        return None
    return [name.strip() for name in value.split(',')]


def validate_export_params(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Validate the combination of export parameters."""
    version = ctx.params.get('version')
    model_dir = ctx.params.get('model_dir')
    model_filename = ctx.params.get('model_filename')
    params_filename = ctx.params.get('params_filename')
    weights = ctx.params.get('weights')

    if version == 'pp-yoloe':
        if not all([model_dir, model_filename, params_filename]):
            raise click.BadParameter('For PP-YOLOE, --model_dir, --model_filename and --params_filename are required')
    elif version and version != 'pp-yoloe':
        if not weights:
            raise click.BadParameter('For non PP-YOLOE models, --weights is required')

    return value


@click.group()
def trtyolo():
    """Command line tool for exporting models and performing inference with TensorRT-YOLO."""
    pass


@trtyolo.command(
    help="Export YOLO models to ONNX format compatible with TensorRT-YOLO. Supports YOLOv3, YOLOv5, YOLOv8, YOLOv10, YOLO11, YOLO12, YOLO-World, YOLOE, PP-YOLOE, and PP-YOLOE+."
)
@click.option(
    '-v',
    '--version',
    help='Model version. Options include yolov3, yolov5, yolov8, yolov10, yolo11, yolo12, yolo-world, yoloe, pp-yoloe, ultralytics.',
    type=str,
    required=True,
    callback=validate_export_params,
)
@click.option(
    '-o',
    '--output',
    help='Directory path to save the exported model.',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    required=True,
)
@click.option(
    '-w',
    '--weights',
    help='Path to PyTorch YOLO weights (required for non PP-YOLOE models).',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
)
@click.option(
    '--model_dir',
    help='Directory path containing the PaddleDetection PP-YOLOE model (required for PP-YOLOE).',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
)
@click.option('--model_filename', help='Filename of the PaddleDetection PP-YOLOE model (required for PP-YOLOE).', type=str)
@click.option('--params_filename', help='Filename of the PaddleDetection PP-YOLOE parameters (required for PP-YOLOE).', type=str)
@click.option('-b', '--batch', default=1, help='Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.', type=int)
@click.option('--max_boxes', default=100, help='Maximum number of detections per image. Defaults to 100.', type=int)
@click.option('--iou_thres', default=0.45, help='NMS IoU threshold for post-processing. Defaults to 0.45.', type=float)
@click.option('--conf_thres', default=0.25, help='Confidence threshold for object detection. Defaults to 0.25.', type=float)
@click.option(
    '--imgsz',
    default='640',
    help='Image size (single value for square or "height,width"). Defaults to "640" (for non PP-YOLOE models).',
    type=str,
    callback=validate_imgsz,
)
@click.option(
    '-n',
    '--names',
    help='Custom class names for YOLO-World and YOLOE (comma-separated, e.g., "person,car,dog"). Only applicable for YOLO-World and YOLOE models.',
    type=str,
    callback=validate_names,
)
@click.option(
    '--repo_dir',
    help='Directory containing the local repository (if using torch.hub.load). Only applicable for YOLOv3 and YOLOv5 models.',
    type=str,
)
@click.option('--opset', default=12, help='ONNX opset version. Defaults to 12.', type=int)
@click.option('-s', '--simplify', is_flag=True, help='Whether to simplify the exported ONNX model. Defaults to False.')
def export(
    version: str,
    output: str,
    weights: Optional[str],
    model_dir: Optional[str],
    model_filename: Optional[str],
    params_filename: Optional[str],
    imgsz: Tuple[int, int],
    names: Optional[List[str]],
    repo_dir: Optional[str],
    batch: int,
    max_boxes: int,
    iou_thres: float,
    conf_thres: float,
    opset: int,
    simplify: bool,
):
    """Export models for TensorRT-YOLO.

    This command allows exporting models for both PaddlePaddle and PyTorch frameworks to be used with TensorRT-YOLO.
    """
    from .export import paddle_export, torch_export

    if version == 'pp-yoloe':
        paddle_export(
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch=batch,
            output=output,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset,
            simplify=simplify,
        )
    else:
        torch_export(
            weights=weights,
            output=output,
            version=version,
            imgsz=imgsz,
            batch=batch,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset,
            simplify=simplify,
            repo_dir=repo_dir,
            custom_classes=names,
        )

    logger.success("Export completed successfully!")


@trtyolo.command(help="Perform inference with TensorRT-YOLO.")
@click.option(
    '-e',
    '--engine',
    help='Engine file for inference.',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
    required=True,
)
@click.option(
    '-m',
    '--mode',
    help='Mode for inference: 0 for Classify, 1 for Detect, 2 for OBB, 3 for Segment, 4 for Pose.',
    type=click.IntRange(0, 4),
    required=True,
)
@click.option(
    '-i',
    '--input',
    help='Input directory or file for inference.',
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=str),
    required=True,
)
@click.option(
    '-o', '--output', help='Output directory for inference results.', type=click.Path(file_okay=False, dir_okay=True, path_type=str)
)
@click.option(
    '-l', '--labels', help='Labels file for inference.', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str)
)
def infer(engine: str, mode: int, input: str, output: str, labels: str):
    """Perform inference with TensorRT-YOLO.

    This command performs inference using TensorRT-YOLO with the specified engine file and input source.
    """
    if output and not labels:
        logger.error("Please provide a labels file using -l or --labels.")
        raise click.Abort()

    if output:
        from .infer import generate_labels

        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        labels = generate_labels(labels)

    import cv2
    from rich.progress import track

    from .infer import (
        ClassifyModel,
        DetectModel,
        InferOption,
        OBBModel,
        PoseModel,
        SegmentModel,
        image_batches,
        visualize,
    )

    option = InferOption()
    option.enable_swap_rb()
    option.enable_performance_report()

    model = (
        DetectModel(engine, option)
        if mode == 1
        else OBBModel(engine, option)
        if mode == 2
        else SegmentModel(engine, option)
        if mode == 3
        else PoseModel(engine, option)
        if mode == 4
        else ClassifyModel(engine, option)
    )

    batchs = image_batches(input, model.batch_size, True)

    logger.info(f"Infering data in {input}")
    for batch in track(batchs, description="[cyan]Processing batches", total=len(batchs)):
        images = [cv2.imread(image_path) for image_path in batch]
        results = model.predict(images)

        if output:
            for image_path, image, result in zip(batch, images, results):
                vis_image = visualize(image, result, labels)
                cv2.imwrite(str(output_dir / Path(image_path).name), vis_image)

    logger.success("Finished Inference.")
    model.performance_report()


if __name__ == '__main__':
    trtyolo()
