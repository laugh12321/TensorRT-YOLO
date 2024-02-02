#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
# Version :   2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/01/26 16:55:22
# Desc    :   This script exports a YOLOv5 model to TensorRT engine.
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

import time
import warnings
from copy import deepcopy

import torch
from nms import Detect

from ultralytics.utils.files import file_size
from ultralytics.engine.exporter import try_export
from ultralytics.utils import LOGGER, colorstr, __version__, LINUX
from ultralytics.nn.autobackend import check_class_names, default_class_names
from ultralytics.utils.torch_utils import get_latest_opset, smart_inference_mode
from ultralytics.utils.checks import check_imgsz, check_requirements, check_version


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export YOLOv5 model to TensorRT engine.')
    parser.add_argument('-w', '--weights', required=True, help='Path to Ultralytics YOLOv5 model weights file.')
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


class Exporter:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    @smart_inference_mode()
    def __call__(self, model=None) -> None:
        t = time.time()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Checks
        if not hasattr(model, "names"):
            model.names = default_class_names()
        model.names = check_class_names(model.names)
        if self.args.precision == 'fp16' and self.device.type == "cpu":
            LOGGER.warning("WARNING ⚠️ FP16 only compatible with GPU export, i.e. use device=0")
            self.args.precision == 'fp32'
        self.imgsz = check_imgsz(self.args.imgsz, stride=model.stride, min_dim=2)  # check image size

        # Input
        im = torch.zeros(self.args.batch, 3, *self.imgsz).to(self.device)

        # Update model
        model = deepcopy(model).to(self.device)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.float()
        for m in model.modules():
            if m.__class__.__name__ == 'Detect':
                detect = Detect
                detect.inplace = False
                detect.dynamic = False
                detect.export = True
                detect.conf_thres = self.args.conf_thres
                detect.iou_thres = self.args.iou_thres
                detect.max_det = self.args.max_boxes
                setattr(m, '__class__', detect)

        for _ in range(2):
            model(im)  # dry runs

        # Filter warnings
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model

        LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {self.args.weights} ({file_size(self.args.weights):.1f} MB)")

        # Exports
        f, _ = self.export_engine() # TensorRT required before ONNX

        square = self.imgsz[0] == self.imgsz[1]
        s = (
            ""
            if square
            else f"WARNING ⚠️ non-PyTorch val requires square images, 'imgsz={self.imgsz}' will not work."
        )
        imgsz = self.imgsz[0] if square else str(self.imgsz)[1:-1].replace(" ", "")
        LOGGER.info(
            f'\nExport complete ({time.time() - t:.1f}s)'
            f"\nResults saved to {colorstr('bold', self.args.output)}"
            f'\nDetails:         yolo model={f} imgsz={imgsz} {s}'
            f'\nVisualize:       https://netron.app'
        )

    @try_export
    def export_onnx(self, prefix=colorstr("ONNX:")):
        # YOLOv5 ONNX export
        requirements = ["onnx>=1.12.0"]
        if self.args.simplify:
            requirements += ["onnxsim>=0.4.33", "onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime"]
        check_requirements(requirements)
        import onnx  # noqa

        opset_version = self.args.opset or get_latest_opset()
        LOGGER.info(f"\n{prefix} starting export with onnx {onnx.__version__} opset {opset_version}...")

        f = str(Path(self.args.output, Path(self.args.weights).stem).with_suffix(".onnx"))

        output_names = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']

        torch.onnx.export(
            self.model,  # dynamic=True only compatible with cpu
            self.im,
            f,
            verbose=False,
            opset_version=opset_version,
            do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=None,
        )

        # Checks
        model_onnx = onnx.load(f)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        shapes = {
            'num_detections': [self.args.batch, 1],
            'detection_boxes': [self.args.batch, self.args.max_boxes, 4],
            'detection_scores': [self.args.batch, self.args.max_boxes],
            'detection_classes': [self.args.batch, self.args.max_boxes]
        }
        for output in model_onnx.graph.output:
            for idx, dim in enumerate(output.type.tensor_type.shape.dim):
                dim.dim_param = str(shapes[output.name][idx])

        # Simplify
        if self.args.simplify:
            try:
                import onnxsim

                LOGGER.info(f"{prefix} simplifying with onnxsim {onnxsim.__version__}...")
                # subprocess.run(f'onnxsim "{f}" "{f}"', shell=True)
                model_onnx, check = onnxsim.simplify(model_onnx)
                assert check, "Simplified ONNX model could not be validated"
            except Exception as e:
                LOGGER.info(f"{prefix} simplifier failure: {e}")

        onnx.save(model_onnx, f)
        return f, model_onnx


    @try_export
    def export_engine(self, prefix=colorstr("TensorRT:")):
        """YOLOv8 TensorRT export https://developer.nvidia.com/tensorrt."""
        # assert self.im.device.type != "cpu", "export running on CPU but must be on GPU, i.e. use 'device=0'"
        f_onnx, _ = self.export_onnx()  # run before trt import https://github.com/ultralytics/ultralytics/issues/7016

        try:
            import tensorrt as trt  # noqa
        except ImportError:
            if LINUX:
                check_requirements("nvidia-tensorrt", cmds="-U --index-url https://pypi.ngc.nvidia.com")
            import tensorrt as trt  # noqa

        check_version(trt.__version__, "7.0.0", hard=True)  # require tensorrt>=7.0.0

        LOGGER.info(f"\n{prefix} starting export with TensorRT {trt.__version__}...")
        assert Path(f_onnx).exists(), f"failed to export ONNX file: {f_onnx}"
        f = str(Path(self.args.output, Path(self.args.weights).stem).with_suffix(".engine"))
        logger = trt.Logger(trt.Logger.INFO)
        if self.args.verbose:
            logger.min_severity = trt.Logger.Severity.VERBOSE

        trt.init_libnvinfer_plugins(logger, namespace="")

        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = self.args.workspace * 1 << 30
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace << 30)  # fix TRT 8.4 deprecation notice

        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(f_onnx):
            raise RuntimeError(f"failed to load ONNX file: {f_onnx}")

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            LOGGER.info(f'{prefix} input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        for out in outputs:
            LOGGER.info(f'{prefix} output "{out.name}" with shape{out.shape} {out.dtype}')

        LOGGER.info(
            f"{prefix} building FP{16 if builder.platform_has_fast_fp16 and self.args.precision == 'fp16' else 32} engine as {f}"
        )
        if builder.platform_has_fast_fp16 and self.args.precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        del self.model
        torch.cuda.empty_cache()

        # Write file
        with builder.build_serialized_network(network, config) as engine, open(f, "wb") as t:
            t.write(engine)
            LOGGER.info(f"Serialize engine success, saved as {f}")

        return f, None


if __name__ == '__main__':
    opt = parse_opt()
    # Load model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=opt.weights)
    Exporter(opt)(model)
