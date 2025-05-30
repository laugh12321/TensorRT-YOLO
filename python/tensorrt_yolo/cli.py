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
# Version :   6.0.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:26:53
# Desc    :   trtyolo cli.
# ==============================================================================
import sys
from pathlib import Path

import rich_click as click
from loguru import logger

logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])


@click.group()
def trtyolo():
    """Command line tool for exporting models and performing inference with TensorRT-YOLO."""
    pass


@trtyolo.command(help="Export models for TensorRT-YOLO. Supports YOLOv3, YOLOv5, YOLOv8, YOLOv10, YOLO11, YOLO12, PP-YOLOE and PP-YOLOE+.")
@click.option('--model_dir', help='Path to the directory containing the PaddleDetection PP-YOLOE model.', type=str)
@click.option('--model_filename', help='The filename of the PP-YOLOE model.', type=str)
@click.option('--params_filename', help='The filename of the PP-YOLOE parameters.', type=str)
@click.option('-w', '--weights', help='Path to YOLO weights for PyTorch.', type=str)
@click.option('-v', '--version', help='Torch YOLO version, e.g., yolov3, yolov5, yolov8, yolov10, yolo11, yolo12, ultralytics.', type=str)
@click.option('--imgsz', nargs=2, default=[640, 640], help='Image size (height, width). Defaults to [640, 640].', type=int)
@click.option('--repo_dir', default=None, help='Directory containing the local repository (if using torch.hub.load).', type=str)
@click.option('-o', '--output', help='Directory path to save the exported model.', type=str, required=True)
@click.option('-b', '--batch', default=1, help='Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.', type=int)
@click.option('--max_boxes', default=100, help='Maximum number of detections to output per image. Defaults to 100.', type=int)
@click.option('--iou_thres', default=0.45, help='NMS IoU threshold for post-processing. Defaults to 0.45.', type=float)
@click.option('--conf_thres', default=0.25, help='Confidence threshold for object detection. Defaults to 0.25.', type=float)
@click.option('--opset_version', default=12, help='ONNX opset version. Defaults to 12.', type=int)
@click.option('-s', '--simplify', is_flag=True, help='Whether to simplify the exported ONNX. Defaults is False.')
def export(
    model_dir,
    model_filename,
    params_filename,
    weights,
    version,
    imgsz,
    repo_dir,
    output,
    batch,
    max_boxes,
    iou_thres,
    conf_thres,
    opset_version,
    simplify,
):
    """Export models for TensorRT-YOLO.

    This command allows exporting models for both PaddlePaddle and PyTorch frameworks to be used with TensorRT-YOLO.
    """

    from .export import paddle_export, torch_export

    if model_dir and model_filename and params_filename:
        paddle_export(
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch=batch,
            output=output,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset_version,
            simplify=simplify,
        )
    elif weights and version:
        torch_export(
            weights=weights,
            output=output,
            version=version,
            imgsz=imgsz,
            batch=batch,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset_version,
            simplify=simplify,
            repo_dir=repo_dir,
        )
    else:
        logger.error("Please provide correct export parameters.")


@trtyolo.command(help="Perform inference with TensorRT-YOLO.")
@click.option('-e', '--engine', help='Engine file for inference.', type=str, required=True)
@click.option(
    '-m', '--mode', help='Mode for inference: 0 for Classify, 1 for Detect, 2 for OBB, 3 for Segment, 4 for Pose.', type=int, required=True
)
@click.option('-i', '--input', help='Input directory or file for inference.', type=str, required=True)
@click.option('-o', '--output', help='Output directory for inference results.', type=str)
@click.option('-l', '--labels', help='Labels file for inference.', type=str)
def infer(engine, mode, input, output, labels):
    """Perform inference with TensorRT-YOLO.

    This command performs inference using TensorRT-YOLO with the specified engine file and input source.
    """
    if mode not in (0, 1, 2, 3, 4):
        logger.error(f"Invalid mode: {mode}. Please use 0 for Classify, 1 for Detect, 2 for OBB, 3 for Segment, 4 for Pose.")
        sys.exit(1)

    if output and not labels:
        logger.error("Please provide a labels file using -l or --labels.")
        sys.exit(1)

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
