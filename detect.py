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
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/29 15:13:41
# Desc    :   YOLO Series Inference Script.
# ==============================================================================
import cv2
import argparse
from pathlib import Path
from python.infer.yolo import YOLO
from python.utils import visualize

IMG_FORMATS = [".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"]


def parse_opt() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='YOLO Series Inference Script.')
    parser.add_argument('-w', '--weights', required=True, type=str, help='Path to TensorRT engine file.')
    parser.add_argument('-s', '--source', required=True, type=str, help="Path to input image or directory.")
    parser.add_argument('-o', '--output', type=str, default=None, help='Directory path to save the output images.')
    parser.add_argument('--max-image-size', nargs='+', type=int, default=[1080, 1920], help='Maximum inference image size (height, width).')
    parser.add_argument('--benchmark', action='store_true', help='Enable benchmarking.')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    from functools import reduce
    from operator import mul

    opt = parse_opt()

    source = Path(opt.source)
    if opt.output is not None:
        output = Path(opt.output)
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = None

    # Load model
    model = YOLO(opt.weights, reduce(mul, opt.max_image_size))
    model.warmup()

    # Inference
    if source.is_file():
        image = cv2.imread(str(source))
        det_info = model.infer(image)
        if output is not None:
            vis_image = visualize(image, det_info)
            cv2.imwrite(str(output / source.name), vis_image)
    elif source.is_dir():
        files = [file for file in list(source.glob('**/*.*')) if file.suffix.lower() in IMG_FORMATS]
        for idx in range(0, len(files), model.batch_size):
            images = [cv2.imread(str(files[idx+i])) for i in range(model.batch_size) if idx+i < len(files)]
            det_infos = model.batch_infer(images, opt.benchmark)
            if output is not None:
                for i, image in enumerate(images):
                    vis_image = visualize(image, det_infos[i])
                    cv2.imwrite(str(output / files[idx+i].name), vis_image)
        if opt.benchmark: model.benchmark.info()

    print("Finished.")