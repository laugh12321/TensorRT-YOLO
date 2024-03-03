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
# Date    :   2024/03/03 09:45:18
# Desc    :   This script exports a YOLOv9 model to ONNX.
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
from common import Detect, DDetect, DualDetect, DualDDetect

from ultralytics.utils.files import file_size
from ultralytics.engine.exporter import try_export
from ultralytics.utils import LOGGER, colorstr, __version__
from ultralytics.utils.checks import check_imgsz, check_requirements
from ultralytics.utils.torch_utils import get_latest_opset, smart_inference_mode

DETECT = {
    'Detect': Detect, 
    'DDetect': DDetect, 
    'DualDetect': DualDetect, 
    'DualDDetect': DualDDetect,
}

class Exporter:
    """
    A class for exporting a model.

    Ref: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/engine/exporter.py#L141

    Attributes:
        args (argparse.Namespace): Configuration for the exporter.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Initializes the Exporter class.

        Args:
            args (argparse.Namespace): Configuration for the exporter.
        """
        self.args = args

    @smart_inference_mode()
    def __call__(self, model=None) -> None:
        """Returns exported file."""
        t = time.time()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Checks
        if self.args.half and self.device.type == "cpu":
            LOGGER.warning("WARNING ⚠️ FP16 only compatible with GPU export, i.e. use device=0")
            self.args.half = False
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
            # print(m.__class__.__name__)
            if m.__class__.__name__ in DETECT.keys():
                detect = DETECT[m.__class__.__name__]
                detect.inplace = False
                detect.dynamic = False
                detect.export = True
                detect.half = self.args.half
                detect.conf_thres = self.args.conf_thres
                detect.iou_thres = self.args.iou_thres
                detect.max_det = self.args.max_boxes
                setattr(m, '__class__', detect)

        for _ in range(2):
            model(im)  # dry runs
        if self.args.half:
            im, model = im.half(), model.half()  # to FP16

        # Filter warnings
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)  # suppress TracerWarning
        warnings.filterwarnings("ignore", category=UserWarning)  # suppress shape prim::Constant missing ONNX warning
        warnings.filterwarnings("ignore", category=DeprecationWarning)  # suppress CoreML np.bool deprecation warning

        # Assign
        self.im = im
        self.model = model
        LOGGER.info(f"\n{colorstr('PyTorch:')} starting from {self.args.weights} ({file_size(self.args.weights):.1f} MB)")

        # Exports
        f, _ = self.export_onnx()

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

        return f  # return exported file

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


def parse_opt() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Export YOLOv5 model to ONNX.')
    parser.add_argument('-w', '--weights', required=True, help='Path to Ultralytics YOLOv5 model weights file.')
    parser.add_argument('-o', '--output', required=True, type=str, help='Directory path to save the exported model.')
    parser.add_argument('-b', '--batch', type=int, default=1, help='Total batch size for the model.')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='Inference size (height, width).')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold for object detection.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold for post-processing.')
    parser.add_argument('--max-boxes', type=int, default=100, help='Maximum number of detections to output per image.')
    parser.add_argument('-s', '--simplify', action='store_true', help='Whether to simplify the exported ONNX. Default is False.')
    parser.add_argument("--half", action="store_true", help="FP16 half-precision export")
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version.')

    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # Expand imgsz if only one value is provided

    return opt


if __name__ == '__main__':
    opt = parse_opt()

    # Load a model
    model = torch.hub.load('WongKinYiu/yolov9', 'custom', path=opt.weights)

    # Export ONNX model
    Exporter(opt)(model)
