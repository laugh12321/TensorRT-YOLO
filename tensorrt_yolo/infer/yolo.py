#!/usr/bin/env python
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
# Version :   3.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/28 22:13:39
# Desc    :   CUDA-Python Inference.
# ==============================================================================
from time import perf_counter_ns
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorrt as trt
from cuda import cudart
from loguru import logger
from polygraphy.backend.common import BytesFromPath
from polygraphy.backend.trt import EngineFromBytes
from polygraphy.util import invoke_if_callable, is_shape_dynamic

from .common import HostDeviceMem, cuda_assert
from .general import scale_boxes
from .structs import DetectInfo, TensorInfo

__all__ = ['TRTYOLO']


# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
# If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
def allocate_buffers(engine: trt.ICudaEngine):
    tensors = []
    stream = cuda_assert(cudart.cudaStreamCreate())
    dynamic = False  # Initialize dynamic flag
    batch = 1  # Initialize batch size
    input_dtype = None

    for binding in engine:
        # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
        # Pick out the max shape to allocate enough memory for the binding.
        input = engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT
        shape = engine.get_tensor_shape(binding)
        if dynamic := is_shape_dynamic(shape):
            if input:
                # Get the maximum profile shape
                shape = engine.get_tensor_profile_shape(binding, 0)[-1]
                batch = shape[0]  # Set batch size
            else:
                # Assuming batch size has already been set
                shape = shape[1:]

        size = trt.volume(shape)
        dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))

        if input:
            input_dtype = dtype
            batch, _, *imgsz = shape

        # Allocate host and device buffers
        bindingMemory = HostDeviceMem(size, dtype)

        # Append to the tensors list.
        tensors.append(
            TensorInfo(
                name=binding,
                shape=shape,
                input=input,
                memory=bindingMemory,
            )
        )

    return tensors, dynamic, stream, batch, imgsz, input_dtype


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
        self._engine, _ = invoke_if_callable(EngineFromBytes(BytesFromPath(engine_path)))
        self._tensors, self._dynamic, self._stream, self._batch_size, self._imgsz, self._dtype = allocate_buffers(self._engine)

    def destory(self) -> None:
        """
        Clean up resources when the instance is deleted.
        """
        for tensor in self._tensors:
            tensor.memory.free()
        cuda_assert(cudart.cudaStreamDestroy(self._stream))
        del self._engine

    def _infer(self) -> None:
        """
        Run inference on the TensorRT engine.
        """
        # Transfer input data to the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [
            cuda_assert(cudart.cudaMemcpyAsync(tensor.memory.device, tensor.memory.host, tensor.memory.nbytes, kind, self._stream))
            for tensor in self._tensors
            if tensor.input
        ]

        # Run inference.
        with self._engine.create_execution_context() as context:
            for tensor in self._tensors:
                context.set_tensor_address(tensor.name, int(tensor.memory.device))
                if tensor.input and self._dynamic:
                    context.set_input_shape(tensor.name, tensor.shape)
        context.execute_async_v3(self._stream)

        # Transfer predictions back from the GPU.
        kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [
            cuda_assert(cudart.cudaMemcpyAsync(tensor.memory.host, tensor.memory.device, tensor.memory.nbytes, kind, self._stream))
            for tensor in self._tensors
            if not tensor.input
        ]

        # Synchronize the stream
        cuda_assert(cudart.cudaStreamSynchronize(self._stream))

    def _postprocess(self, outputs: Dict[str, np.ndarray], output_shape: Tuple[int, int], idx: Optional[int] = 0) -> DetectInfo:
        """
        Process the inference results.

        Args:
            outputs (Dict[str, np.ndarray]): Dictionary containing output tensors.
            output_shape (Tuple[int, int]): output image shape.
            idx (Optional[int], optional): Index of the batch. Defaults to 0.

        Returns:
            DetectInfo: Processed detection information.
        """
        num_detections = int(outputs['num_dets'][idx])
        detection_boxes = outputs['det_boxes'][idx, :num_detections]
        detection_scores = outputs['det_scores'][idx, :num_detections]
        detection_classes = outputs['det_classes'][idx, :num_detections]

        detection_boxes = scale_boxes(detection_boxes, self._imgsz, output_shape)

        return DetectInfo(
            num=num_detections,
            boxes=detection_boxes,
            scores=detection_scores,
            classes=detection_classes,
        )

    @property
    def batch_size(self) -> int:
        """
        Get the batch size.

        Returns:
            int: The batch size used for inference.
        """
        return self._batch_size

    @property
    def imgsz(self) -> List[int]:
        """
        Get the image size.

        Returns:
            List[int]: A list containing the height and width of the input images.
        """
        return self._imgsz

    @property
    def dtype(self) -> np.dtype:
        """
        Get the data type of the tensor.

        Returns:
            np.dtype: The data type of the tensor.
        """
        return self._dtype

    @property
    def dynamic(self) -> bool:
        """
        Check if the tensor is dynamic.

        Returns:
            bool: True if the tensor is dynamic, False otherwise.
        """
        return self._dynamic

    def warmup(self, iters: int = 10) -> None:
        """
        Warm up the TensorRT engine by running a specified number of inference iterations.

        Args:
            iters (int, optional): Number of warm-up iterations. Defaults to 10.
        """
        start_time_ns = perf_counter_ns()
        for _ in range(iters):
            self._infer()
        end_time_ns = perf_counter_ns()
        elapsed_time_ms = (end_time_ns - start_time_ns) / 1e6
        logger.info(f"warmup {iters} iters cost {elapsed_time_ms:.2f} ms.")

    def infer(self, batch: np.ndarray, batch_shape: List[Tuple[int, int]]) -> List[DetectInfo]:
        """
        Run inference on the TensorRT engine.

        Args:
            batch (np.ndarray): Input batch for inference.
            batch_shape (List[Tuple[int, int]]): image shape for each batch item.

        Returns:
            List[DetectInfo]: List of processed detection information for each batch item.
        """
        for tensor in self._tensors:
            if tensor.input:
                tensor.memory.host = batch
                tensor.shape = batch.shape
                batch_size = batch.shape[0]
            else:
                tensor.shape[0] = batch_size

        # Run inference
        self._infer()

        # Get only the host outputs.
        outputs = {
            tensor.name: tensor.memory.host[: np.prod(tensor.shape)].reshape(tensor.shape) for tensor in self._tensors if not tensor.input
        }

        # Process the outputs
        return [self._postprocess(outputs, shape, idx) for idx, shape in enumerate(batch_shape)]
