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
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/28 22:13:39
# Desc    :   PyCUDA Inference.
# ==============================================================================
import os
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from typing import List, Dict, Tuple

import cv2
import time
import numpy as np

import tensorrt as trt
import pycuda.autoinit
from pycuda import gpuarray
import pycuda.driver as driver
from pycuda.compiler import SourceModule
from pycuda.tools import DeviceMemoryPool

from python.utils import Timer, DetectInfo, TensorInfo

__all__ = ['YOLO']


class YOLO:

    def __init__(self, engine_path: str, max_image_size: int = 1080*1920) -> None:
        self._init_variables()
        self._init_model(engine_path)
        self._init_cuda_kernel()
        self._setup_variables(max_image_size)

        # for benchmark
        self.benchmark = Timer(self._batch_size)

    def _init_variables(self) -> None:
        self._pool = DeviceMemoryPool()
        self._stream, self._context = None, None
        self._bindings, self._tensors = [], []
        self._d2s_device, self._image_device = None, None

    def _init_cuda_kernel(self) -> None:
        # preprocess kernel
        preprocess = SourceModule(open(ROOT / 'preprocess.cu', encoding='utf-8').read(), no_extern_c=True)
        if np.issubdtype(self._tensors[0].dtype, np.float16):
            self._preprocess_kernel = preprocess.get_function("preprocess_kernel_fp16")
        else:
            self._preprocess_kernel = preprocess.get_function("preprocess_kernel_fp32")

    def _init_model(self, engine_path: str) -> None:
        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, namespace="")

        # Load TRT engine
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            assert runtime
            engine = runtime.deserialize_cuda_engine(f.read())
        assert engine

        self._stream = driver.Stream()
        self._context = engine.create_execution_context()

        # Setup I/O bindings
        for binding in engine:
            shape = tuple(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host = driver.pagelocked_empty(shape, dtype)
            device = gpuarray.to_gpu_async(host, allocator=self._pool.allocate, stream=self._stream)

            # Append to tensors
            self._bindings.append(device.ptr)
            self._tensors.append(TensorInfo(binding, shape, dtype, host, device))

    def _setup_variables(self, max_image_size: int) -> None:
        self._batch_size, _, *self._image_size = self._tensors[0].shape
        self._d2s_host = driver.pagelocked_empty((2, 3), dtype=np.float32)
        self._d2s_device = gpuarray.to_gpu_async(self._d2s_host, allocator=self._pool.allocate, stream=self._stream)
        self._image_host = driver.pagelocked_empty(max_image_size*3, dtype=np.uint8)
        self._image_device = gpuarray.to_gpu_async(self._image_host, allocator=self._pool.allocate, stream=self._stream)

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def image_size(self) -> Tuple[int, int]:
        return tuple(self._image_size)

    def __del__(self):
        self._pool.stop_holding()

        if self._tensors:
            [tensor.device.gpudata.free() for tensor in self._tensors]

        if self._image_device is not None:
            self._image_device.gpudata.free()

        if self._d2s_device is not None:
            self._d2s_device.gpudata.free()

        del self._tensors
        del self._stream
        del self._context

    def warmup(self, iters: int = 10) -> None:
        start_time_ns = time.perf_counter_ns()
        for _ in range(iters):
            self._tensors[0].host.fill(np.random.randint(0, 255))
            self._tensors[0].device.set_async(self._tensors[0].host, self._stream)
            self._infer()
        end_time_ns = time.perf_counter_ns()
        elapsed_time_ms = (end_time_ns - start_time_ns) / 1e6
        print(f"warmup {iters} iters cost {elapsed_time_ms:.2f} ms.")

    def infer(self, image: np.ndarray) -> DetectInfo:
        ratio = self._preprocess(image, self._tensors[0].device)
        self._infer()
        infer = self._postprocess()
        height, width = image.shape[:2]
        x_offset = round((self._image_size[0] * ratio - width) / 2)
        y_offset = round((self._image_size[1] * ratio - height) / 2)
        return self._process_detection(infer, height, width, ratio, x_offset, y_offset)

    def batch_infer(self, images: List[np.ndarray], run_benchmark: bool = False) -> List[DetectInfo]:
        ratios = []
        if run_benchmark:
            self.benchmark.infer_count += 1
            self.benchmark.pre_time.start()
        for idx, image in enumerate(images):
            ratios.append(self._preprocess(image, self._tensors[0].device[idx]))
        if run_benchmark:
            self.benchmark.pre_time.end()

        if run_benchmark:
            self.benchmark.infer_time.start()
        self._infer()
        if run_benchmark:
            self.benchmark.infer_time.end()

        if run_benchmark:
            self.benchmark.post_time.start()
        infer = self._postprocess()
        detections = []
        for idx, image in enumerate(images):
            height, width = image.shape[:2]
            x_offset = round((self._image_size[0] * ratios[idx] - width) / 2)
            y_offset = round((self._image_size[1] * ratios[idx] - height) / 2)
            detections.append(self._process_detection(infer, height, width, ratios[idx], x_offset, y_offset, idx))
        if run_benchmark:
            self.benchmark.post_time.end()

        return detections

    def _process_detection(self, infer: Dict[str, np.ndarray], height: int, width: int, ratio: float, x_offset: float, y_offset: float, idx: int = 0) -> DetectInfo:
        num_detections = int(infer['num_detections'][idx])
        detection_boxes = infer['detection_boxes'][idx, :num_detections]

        detection_boxes[:, 0] = np.maximum(detection_boxes[:, 0] * ratio - x_offset, 0.0)
        detection_boxes[:, 1] = np.maximum(detection_boxes[:, 1] * ratio - y_offset, 0.0)
        detection_boxes[:, 2] = np.minimum(detection_boxes[:, 2] * ratio - x_offset, width)
        detection_boxes[:, 3] = np.minimum(detection_boxes[:, 3] * ratio - y_offset, height)

        detection_scores = infer['detection_scores'][idx, :num_detections]
        detection_classes = infer['detection_classes'][idx, :num_detections]

        return DetectInfo(
            num=num_detections,
            boxes=detection_boxes,
            scores=detection_scores,
            classes=detection_classes,
        )

    def _preprocess(self, image: np.ndarray, device: gpuarray) -> float:
        height, width = map(np.int32, image.shape[:2])
        dst_width, dst_height = map(np.int32, self._image_size)

        r = min(self._image_size[1] / height, self._image_size[0] / width)

        s2d = np.array([[r, 0, -r * width * 0.5 + dst_width * 0.5],
                        [0, r, -r * height * 0.5 + dst_height * 0.5]], dtype=np.float32)
        cv2.invertAffineTransform(s2d, self._d2s_host)

        np.copyto(self._image_host[:image.size], image.ravel())
        self._d2s_device.set_async(self._d2s_host, stream=self._stream)
        self._image_device.set_async(self._image_host, stream=self._stream)

        self._preprocess_kernel(
            self._image_device, width * 3, width, height, 
            device, dst_width, dst_height, 
            np.int32(128), self._d2s_device,
            grid=(32, 32, 1),
            block=(32, 32, 1),
            stream=self._stream
        )

        self._stream.synchronize()

        return 1 / r

    def _infer(self) -> None:
        self._context.execute_async_v2(self._bindings, self._stream.handle)

        for tensor in self._tensors[1:]:
            tensor.device.get_async(self._stream, tensor.host)

        self._stream.synchronize()

    def _postprocess(self) -> Dict[str, np.ndarray]:
        return {tensor.name: tensor.host.reshape(tensor.shape) for tensor in self._tensors[1:]}
