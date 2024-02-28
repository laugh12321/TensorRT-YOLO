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
# File    :   yolo.py
# Version :   2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/28 22:13:39
# Desc    :   PyCUDA Inference.
# ==============================================================================
import time
from typing import List, Tuple

import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

from python.utils import DetectInfo, TensorInfo, scale_boxes

__all__ = ['TRTYOLO']


class TRTYOLO:
    """
    TensorRT YOLO Wrapper

    Args:
        engine_path (str): Path to the TensorRT engine file.
    """
    def __init__(self, engine_path: str) -> None:
        """
        Initialize the TRTYOLO instance.

        Args:
            engine_path (str): Path to the TensorRT engine file.
        """
        self.logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.inputs, self.outputs, self.allocations = self._setup_io_bindings()
        self.stream = cuda.Stream()
        self.width, self.height = self._handle_tensor_shape()

    def _setup_io_bindings(self) -> Tuple[List[TensorInfo], List[TensorInfo], List[cuda.DeviceAllocation]]:
        """
        Setup input and output bindings for the TensorRT engine.

        Returns:
            Tuple[List[TensorInfo], List[TensorInfo], List[cuda.DeviceAllocation]]: Input, output, and allocation information.
        """
        inputs, outputs, allocations = [], [], []
        for binding in self.engine:
            is_input = self.engine.binding_is_input(binding)
            shape = tuple(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            host = cuda.pagelocked_empty(shape, dtype)
            device = cuda.mem_alloc(host.nbytes)

            allocations.append(device)
            tensor = TensorInfo(binding, shape, dtype, host, device)
            inputs.append(tensor) if is_input else outputs.append(tensor)

        return inputs, outputs, allocations

    def _handle_tensor_shape(self) -> Tuple[int, int]:
        """
        Handle the input tensor shape.

        Raises:
            ValueError: If the input shape is invalid.

        Returns:
            Tuple[int, int]: Width and height of the input tensor.
        """
        input_shape = self.inputs[0].shape
        if input_shape[1] == 3:
            height, width = input_shape[2], input_shape[3]
        elif input_shape[3] == 3:
            height, width = input_shape[1], input_shape[2]
        else:
            raise ValueError("Invalid input shape")
        return width, height

    def __del__(self) -> None:
        """
        Clean up resources when the instance is deleted.
        """
        for allocation in self.allocations:
            allocation.free()
        del self.stream
        del self.engine

    def _infer(self) -> None:
        """
        Run inference on the TensorRT engine.
        """
        # Copy I/O and Execute
        for input_tensor in self.inputs:
            cuda.memcpy_htod_async(input_tensor.device, input_tensor.host, self.stream)

        with self.engine.create_execution_context() as context:
            context.execute_async_v2(self.allocations, self.stream.handle)

        for output_tensor in self.outputs:
            cuda.memcpy_dtoh_async(output_tensor.host, output_tensor.device, self.stream)

        self.stream.synchronize()

    def _filter(self, outputs, ratio_pad, idx: int = 0) -> DetectInfo:
        """
        Filter and process the inference results.

        Args:
            outputs (dict): Dictionary containing output tensors.
            ratio_pad (_type_): _description_
            idx (int, optional): Index of the batch. Defaults to 0.

        Returns:
            DetectInfo: Processed detection information.
        """
        num_detections = int(outputs['num_detections'][idx])
        detection_boxes = outputs['detection_boxes'][idx, :num_detections]
        detection_scores = outputs['detection_scores'][idx, :num_detections]
        detection_classes = outputs['detection_classes'][idx, :num_detections]

        detection_boxes = scale_boxes(detection_boxes, (self.height, self.width), ratio_pad)

        return DetectInfo(
            num=num_detections,
            boxes=detection_boxes,
            scores=detection_scores,
            classes=detection_classes,
        )

    def warmup(self, iters: int = 10) -> None:
        """
        Warm up the TensorRT engine by running a specified number of inference iterations.

        Args:
            iters (int, optional): Number of warm-up iterations. Defaults to 10.
        """
        start_time_ns = time.perf_counter_ns()
        for _ in range(iters):
            self._infer()
        end_time_ns = time.perf_counter_ns()
        elapsed_time_ms = (end_time_ns - start_time_ns) / 1e6
        print(f"warmup {iters} iters cost {elapsed_time_ms:.2f} ms.")

    def input_spec(self) -> Tuple[Tuple[int, int, int, int], np.dtype]:
        """
        Get the input tensor specifications.

        Returns:
            Tuple[Tuple[int, int, int, int], np.dtype]: Shape and dtype of the input tensor.
        """    
        return self.inputs[0].shape, self.inputs[0].dtype

    def infer(self, batch, batch_ratio_pad) -> List[DetectInfo]:
        """
        Run inference on the TensorRT engine.

        Args:
            batch (_type_): Input batch for inference.
            batch_ratio_pad (_type_): Ratio_pad for each batch item.

        Returns:
            List[DetectInfo]: List of processed detection information for each batch item.
        """      
        np.copyto(self.inputs[0].host, batch)

        # Run inference
        self._infer()

        # Process the results
        outputs = {tensor.name: tensor.host.reshape(tensor.shape) for tensor in self.outputs}
        return [self._filter(outputs, ratio_pad, idx) for idx, ratio_pad in enumerate(batch_ratio_pad)]
