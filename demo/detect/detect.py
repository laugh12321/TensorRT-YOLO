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
# File    :   detect.py
# Version :   5.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/29 15:13:41
# Desc    :   YOLO Series Inference Script.
# ==============================================================================
import sys
from pathlib import Path
from typing import List

import rich_click as click
from loguru import logger


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


@click.command("YOLO Series Inference Script.")
@click.option('-e', '--engine', required=True, type=str, help='The serialized TensorRT engine.')
@click.option('-m', '--mode', help='Mode for inference: 0 for Detection, 1 for OBB.', type=int, required=True)
@click.option('-i', '--input', required=True, type=str, help="Path to the image or directory to process.")
@click.option('-o', '--output', type=str, default=None, help='Directory where to save the visualization results.')
@click.option("-l", "--labels", default="./labels.txt", help="File to use for reading the class labels from, default: ./labels.txt")
@click.option('--cudaGraph', is_flag=True, help='Optimize inference using CUDA Graphs, compatible with static models only.')
def main(engine, mode, input, output, labels, cudagraph):
    """
    YOLO Series Inference Script.
    """
    import cv2
    from rich.progress import track

    from tensorrt_yolo.infer import CpuTimer, DeployCGDet, DeployDet, GpuTimer, generate_labels_with_colors, visualize_detections

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


if __name__ == '__main__':
    main()
