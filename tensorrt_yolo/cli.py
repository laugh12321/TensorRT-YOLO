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
# Version :   3.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:26:53
# Desc    :   trtyolo cli.
# ==============================================================================
import sys
from pathlib import Path
from typing import List

import rich_click as click
from loguru import logger

logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])


def get_images_in_batches(folder_path: str, batch: int, is_cuda_graph: bool) -> List[List[str]]:
    """
    Get a list of image files in the specified directory, grouped into batches.

    Args:
        folder_path (str): Path to the directory to search for image files.
        batch (int): The number of images in each batch.
        is_cuda_graph (bool): Flag indicating whether to discard extra images if using CUDA graph.

    Returns:
        List[List[str]]: List of image file paths grouped into batches.
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = [path for path in Path(folder_path).rglob('*') if path.suffix.lower() in image_extensions and path.is_file()]

    if not is_cuda_graph:
        # If not using CUDA graph, include all images, padding the last batch if necessary
        batches = [image_files[i : i + batch] for i in range(0, len(image_files), batch)]
    else:
        # If using CUDA graph, exclude extra images that don't fit into a full batch
        total_images = (len(image_files) // batch) * batch
        batches = [image_files[i : i + batch] for i in range(0, total_images, batch)]

    return batches


@click.group()
def trtyolo():
    """Command line tool for exporting models and performing inference with TensorRT-YOLO."""
    pass


@trtyolo.command(help="Export models for TensorRT-YOLO. Supports YOLOv3, YOLOv5, YOLOv6, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, PP-YOLOE and PP-YOLOE+.")
@click.option('--model_dir', help='Path to the directory containing the PaddleDetection PP-YOLOE model.', type=str)
@click.option('--model_filename', help='The filename of the PP-YOLOE model.', type=str)
@click.option('--params_filename', help='The filename of the PP-YOLOE parameters.', type=str)
@click.option('-w', '--weights', help='Path to YOLO weights for PyTorch.', type=str)
@click.option('-v', '--version', help='Torch YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, ultralytics.', type=str)
@click.option('--imgsz', default=640, help='Inference image size. Defaults to 640.', type=int)
@click.option('--repo_dir', default=None, help='Directory containing the local repository (if using torch.hub.load).', type=str)
@click.option('-o', '--output', help='Directory path to save the exported model.', type=str, required=True)
@click.option('-b', '--batch', default=1, help='Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.', type=int)
@click.option('--max_boxes', default=100, help='Maximum number of detections to output per image. Defaults to 100.', type=int)
@click.option('--iou_thres', default=0.45, help='NMS IoU threshold for post-processing. Defaults to 0.45.', type=float)
@click.option('--conf_thres', default=0.25, help='Confidence threshold for object detection. Defaults to 0.25.', type=float)
@click.option('--opset_version', default=11, help='ONNX opset version. Defaults to 11.', type=int)
@click.option('-s', '--simplify', is_flag=True, help='Whether to simplify the exported ONNX. Defaults is True.')
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
@click.option('-m', '--mode', help='Mode for inference: 0 for Detection, 1 for OBB.', type=int, required=True)
@click.option('-i', '--input', help='Input directory or file for inference.', type=str, required=True)
@click.option('-o', '--output', help='Output directory for inference results.', type=str)
@click.option('-l', '--labels', help='Labels file for inference.', type=str)
@click.option('--cudaGraph', help='Optimize inference using CUDA Graphs, compatible with static models only.', is_flag=True)
def infer(engine, mode, input, output, labels, cudagraph):
    """Perform inference with TensorRT-YOLO.

    This command performs inference using TensorRT-YOLO with the specified engine file and input source.
    """

    import cv2
    from rich.progress import track

    from .infer import CpuTimer, DeployCGDet, DeployDet, GpuTimer, generate_labels_with_colors, visualize_detections

    if output:
        output_dir = Path(output)
        output_dir.mkdir(parents=True, exist_ok=True)
        labels = generate_labels_with_colors(labels)

    if mode not in (0, 1):
        logger.error(f"Invalid mode: {mode}. Please use 0 for Detection, 1 for OBB.")
        sys.exit(1)
    is_obb = mode == 1

    if cudagraph:
        model = DeployCGDet(engine, is_obb)
    else:
        model = DeployDet(engine, is_obb)

    cpu_timer = CpuTimer()
    gpu_timer = GpuTimer()

    logger.info(f"Infering data in {input}")

    batchs = get_images_in_batches(input, model.batch, cudagraph)
    for batch in track(batchs, description="[cyan]Processing batches", total=len(batchs)):
        images = [cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2RGB) for image_path in batch]

        cpu_timer.start()
        gpu_timer.start()

        results = model.predict(images)

        cpu_timer.stop()
        gpu_timer.stop()

        if output:
            for image_path, image, result in zip(batch, images, results):
                vis_image = visualize_detections(image, result, labels, is_obb)
                cv2.imwrite(str(output_dir / image_path.name), cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    logger.success(
        "Benchmark results include time for H2D and D2H memory copies, preprocessing, and postprocessing.\n"
        f"    CPU Average Latency: {cpu_timer.milliseconds() / len(batchs):.3f} ms\n"
        f"    GPU Average Latency: {gpu_timer.milliseconds() / len(batchs):.3f} ms\n"
        "    Finished Inference."
    )
