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
# File    :   export.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/28 14:37:02
# Desc    :   This script exports a PP-YOLOE model to TensorRT engine.
# ==============================================================================
import os
import sys
import argparse
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export PP-YOLOE model to TensorRT engine.')
    parser.add_argument('--model_dir', required=True, type=str, help='Path of directory saved PaddleDetection PP-YOLOE model.')
    parser.add_argument('--model_filename', required=True, type=str, help='The PP-YOLOE model file name.')
    parser.add_argument('--params_filename', required=True, type=str, help='The PP-YOLOE parameters file name.')
    parser.add_argument('-o', '--output', required=True, type=str, help='Directory path to save the exported model.')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Total batch size for the model.')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='Inference size (height, width).')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for object detection.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for post-processing.')
    parser.add_argument('--max-boxes', type=int, default=100, help='Maximum number of detections to output per image.')
    parser.add_argument('-p', '--precision', type=str, default='fp32', choices=['fp32', 'fp16'], help="Data type for the engine inference.")
    parser.add_argument('-s', '--simplify', action='store_true', help='Whether to simplify the exported ONNX. Default is False.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--workspace', type=int, default=4, help='Workspace size for TensorRT engine.')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version.')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Expand imgsz if only one value is provided

    return opt


if __name__ == '__main__':
    from trt_helper import export_tensorrt_engine

    opt = parse_opt()

    # Get output file paths
    file_name, file_extension = os.path.splitext(os.path.basename(opt.model_filename))
    onnx_path = os.path.join(opt.output, f'{file_name}.onnx')
    engine_path = os.path.join(opt.output, f'{file_name}.engine')

    # Export TensorRT engine
    export_tensorrt_engine(
        model_dir=opt.model_dir,
        model_filename=opt.model_filename,
        params_filename=opt.params_filename,
        opset=opt.opset,
        batch_size=opt.batch,
        imgsz=opt.imgsz,
        score_thresh=opt.conf_thres,
        nms_thresh=opt.iou_thres,
        onnx_path=onnx_path,
        engine_path=engine_path,
        detections_per_img=opt.max_boxes,
        precision=opt.precision,
        simplify=opt.simplify,
        verbose=opt.verbose,
        workspace=opt.workspace,
    )
