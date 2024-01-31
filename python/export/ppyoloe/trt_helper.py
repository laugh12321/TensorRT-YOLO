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
# File    :   trt_helper.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/28 14:32:04
# Desc    :   Exporting and building TensorRT engines from ONNX graphs.
# ==============================================================================
"""
This code is based on the following repository:
    - https://github.com/zhiqwang/yolort/blob/main/yolort/runtime/trt_helper.py
"""
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from typing import Optional, Tuple

import tensorrt as trt
from trt_graphsurgeon import PPYOLOETRTGraphSurgeon

logging.basicConfig(level=logging.INFO)
logging.getLogger("TRTHelper").setLevel(logging.INFO)
logger = logging.getLogger("TRTHelper")


def export_tensorrt_engine(
    model_dir: str,
    model_filename: str,
    params_filename: str,
    *,
    opset: int = 11,
    batch_size: int = 1,
    imgsz: Tuple[int, int] = (640, 640),
    score_thresh: float = 0.25,
    nms_thresh: float = 0.45,
    onnx_path: Optional[str] = None,
    engine_path: Optional[str] = None,
    detections_per_img: int = 100,
    precision: str = "fp32",
    verbose: bool = False,
    workspace: int = 12,
    simplify: bool = False,
) -> None:
    """
    Export ONNX models and TensorRT serialized engines that can be used for TensorRT inferencing.

    Args:
        model_dir (str): Path of directory saved PaddleDetection PP-YOLOE model.
        model_filename (str): The PP-YOLOE model file name.
        params_filename (str): The PP-YOLOE parameters file name.
        opset (int): ONNX opset version. Default: 11
        batch_size (int): Batch size for inference. Default: 1
        imgsz (Tuple[int, int]): Input image size. Default: (640, 640)
        score_thresh (float): Score threshold used for postprocessing the detections. Default: 0.25
        nms_thresh (float): NMS threshold used for postprocessing the detections. Default: 0.45
        onnx_path (str, optional): The path to the ONNX graph to load. Default: None
        engine_path (str, optional): The path where to serialize the engine to. Default: None
        detections_per_img (int): Number of best detections to keep after NMS. Default: 100
        precision (str): The datatype to use for the engine inference, either 'fp32' or 'fp16'. Default: 'fp32'
        verbose (bool): Enable higher verbosity level for the TensorRT logger. Default: False
        workspace (int): Max memory workspace to allow, in Gb. Default: 12
        simplify (bool, optional): Whether to simplify the exported ONNX. Default to False
    """

    # Set the path of ONNX and Tensorrt Engine to export
    file_name, _ = os.path.splitext(os.path.basename(model_filename))
    onnx_path = onnx_path or os.path.join(model_dir, f'{file_name}.onnx')
    engine_path = engine_path or os.path.join(model_dir, f'{file_name}.engine')

    ppyoloe_gs = PPYOLOETRTGraphSurgeon(
        model_dir=model_dir,
        onnx_path=onnx_path,
        model_filename=model_filename,
        params_filename=params_filename,
        opset=opset,
        batch_size=batch_size,
        imgsz=imgsz,
        simplify=simplify,
    )

    # Register the `EfficientNMS_TRT` into the graph.
    ppyoloe_gs.register_nms(
        score_thresh=score_thresh,
        nms_thresh=nms_thresh,
        detections_per_img=detections_per_img,
    )

    # Save the exported ONNX models.
    ppyoloe_gs.save(onnx_path)

    # Build and export the TensorRT engine.
    engine_builder = EngineBuilder(verbose=verbose, workspace=workspace, precision=precision)
    engine_builder.create_network(onnx_path)
    engine_builder.create_engine(engine_path)


class EngineBuilder:
    """
    Parses an ONNX graph and builds a TensorRT engine from it.
    """

    def __init__(
        self,
        verbose: bool = False,
        workspace: int = 4,
        precision: str = "fp32",
        enable_dynamic: bool = False,
        max_batch_size: int = 16,
        calib_input: Optional[str] = None,
        calib_cache: Optional[str] = None,
        calib_num_images: int = 5000,
        calib_batch_size: int = 8,
    ):
        """
        Args:
            verbose (bool): If enabled, a higher verbosity level will be set on the TensorRT
                logger. Default: False
            workspace (int): Max memory workspace to allow, in Gb. Default: 4
            precision (string): The datatype to use for the engine inference, either 'fp32',
                'fp16' or 'int8'. Default: 'fp32'
            enable_dynamic (bool): Whether to enable dynamic shapes. Default: False
            max_batch_size (int): Maximum batch size reserved for dynamic shape inference.
                Default: 16
            calib_input (string, optinal): The path to a directory holding the calibration images.
                Default: None
            calib_cache (string, optinal): The path where to write the calibration cache to,
                or if it already exists, load it from. Default: None
            calib_num_images (int): The maximum number of images to use for calibration. Default: 5000
            calib_batch_size (int): The batch size to use for the calibration process. Default: 8
        """
        self.logger = trt.Logger(trt.Logger.INFO)
        if verbose:
            self.logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(self.logger, namespace="")

        self.builder = trt.Builder(self.logger)
        self.config = self.builder.create_builder_config()
        self.config.max_workspace_size = workspace * 1 << 30

        self.batch_size = None
        self.network = None
        self.parser = None

        # Leaving some interfaces and parameters for subsequent use, but we have not yet
        # implemented the following functionality
        self.precision = precision
        self.enable_dynamic = enable_dynamic
        self.max_batch_size = max_batch_size
        self.calib_input = calib_input
        self.calib_cache = calib_cache
        self.calib_num_images = calib_num_images
        self.calib_batch_size = calib_batch_size

    def create_network(self, onnx_path: str):
        """
        Parse the ONNX graph and create the corresponding TensorRT network definition.

        Args:
            onnx_path (string): The path to the ONNX graph to load.
        """

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(flag)
        self.parser = trt.OnnxParser(self.network, self.logger)
        if not self.parser.parse_from_file(onnx_path):
            raise RuntimeError(f"Failed to load ONNX file: {onnx_path}")

        inputs = [self.network.get_input(i) for i in range(self.network.num_inputs)]
        outputs = [self.network.get_output(i) for i in range(self.network.num_outputs)]

        logger.info("Network Description")
        for input in inputs:
            self.batch_size = input.shape[0]
            logger.info(f"Input '{input.name}' with shape {input.shape} and dtype {input.dtype}")
        for output in outputs:
            logger.info(f"Output '{output.name}' with shape {output.shape} and dtype {output.dtype}")

    def create_engine(self, engine_path: str):
        """
        Build the TensorRT engine and serialize it to disk.

        Args:
            engine_path (string): The path where to serialize the engine to.
        """
        engine_path = Path(engine_path)
        engine_path.parent.mkdir(parents=True, exist_ok=True)

        precision = self.precision
        logger.info(f"Building {precision} Engine in {engine_path}")

        # Process the batch size and profile
        assert self.batch_size > 0, "Currently only supports static shape."
        self.builder.max_batch_size = self.batch_size

        if precision == "fp16":
            if not self.builder.platform_has_fast_fp16:
                logger.warning("FP16 is not supported natively on this platform/device")
            else:
                self.config.set_flag(trt.BuilderFlag.FP16)
                self.config.set_flag(trt.BuilderFlag.STRICT_TYPES)
        elif precision == "fp32":
            logger.info("Using fp32 mode.")
        else:
            raise NotImplementedError(f"Currently hasn't been implemented: {precision}.")

        with self.builder.build_serialized_network(self.network, self.config) as engine:
            with open(engine_path, "wb") as f:
                f.write(engine)
                logger.info(f"Serialize engine success, saved as {engine_path}")
