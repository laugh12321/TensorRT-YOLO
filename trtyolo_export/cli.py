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
# Version :   6.4.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 14:26:53
# Desc    :   trtyolo cli.
# ==============================================================================
from __future__ import annotations

import sys
import warnings
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple, Union

import onnx
import rich_click as click
import torch
from loguru import logger

# Filter warnings
warnings.filterwarnings("ignore")

# For scripts
logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])


class ModelExporter:
    """Base class for model export functionality."""

    def __init__(self) -> None:
        from .head import (
            UltralyticsClassify,
            UltralyticsDetect,
            UltralyticsOBB,
            UltralyticsPose,
            UltralyticsSegment,
            YOLOClassify,
            YOLODetect,
            YOLOEDetectHead,
            YOLOESegmentHead,
            YOLOSegment,
            YOLOV10Detect,
            YOLOWorldDetect,
        )

        self.__batch = None
        self.__model = None
        self.__head_name = None
        self.__version = None

        self.__max_boxes = None
        self.__dynamic = None
        self.__weight_name = None

        self.__output_names = ["num_dets", "det_boxes", "det_scores", "det_classes"]
        self.__dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "num_dets": {0: "batch"},
            "det_boxes": {0: "batch"},
            "det_scores": {0: "batch"},
            "det_classes": {0: "batch"},
        }
        self.__head_config = {
            "Detect": {
                "class_map": {
                    "yolov3": YOLODetect,
                    "yolov5": YOLODetect,
                    "yolov8": UltralyticsDetect,
                    "yolo11": UltralyticsDetect,
                    "yolo12": UltralyticsDetect,
                    "ultralytics": UltralyticsDetect,
                },
                "output_names": self.__output_names,
                "dynamic_axes": self.__dynamic_axes,
            },
            "v10Detect": {
                "class_map": {"yolov10": YOLOV10Detect, "ultralytics": YOLOV10Detect},
                "output_names": self.__output_names,
                "dynamic_axes": self.__dynamic_axes,
            },
            "WorldDetect": {
                "class_map": {"yolo-world": YOLOWorldDetect, "ultralytics": YOLOWorldDetect},
                "output_names": self.__output_names,
                "dynamic_axes": self.__dynamic_axes,
            },
            "YOLOEDetect": {
                "class_map": {"yoloe": YOLOEDetectHead, "ultralytics": YOLOEDetectHead},
                "output_names": self.__output_names,
                "dynamic_axes": self.__dynamic_axes,
            },
            "YOLOESegment": {
                "class_map": {"yoloe": YOLOESegmentHead, "ultralytics": YOLOESegmentHead},
                "output_names": self.__output_names + ["det_masks"],
                "dynamic_axes": {**self.__dynamic_axes, "det_masks": {0: "batch", 2: "height", 3: "width"}},
            },
            "OBB": {
                "class_map": {
                    "yolov8": UltralyticsOBB,
                    "yolo11": UltralyticsOBB,
                    "yolo12": UltralyticsOBB,
                    "ultralytics": UltralyticsOBB,
                },
                "output_names": self.__output_names,
                "dynamic_axes": self.__dynamic_axes,
            },
            "Segment": {
                "class_map": {
                    "yolov3": YOLOSegment,
                    "yolov5": YOLOSegment,
                    "yolov8": UltralyticsSegment,
                    "yolo11": UltralyticsSegment,
                    "yolo12": UltralyticsSegment,
                    "ultralytics": UltralyticsSegment,
                },
                "output_names": self.__output_names + ["det_masks"],
                "dynamic_axes": {**self.__dynamic_axes, "det_masks": {0: "batch", 2: "height", 3: "width"}},
            },
            "Pose": {
                "class_map": {
                    "yolov8": UltralyticsPose,
                    "yolo11": UltralyticsPose,
                    "yolo12": UltralyticsPose,
                    "ultralytics": UltralyticsPose,
                },
                "output_names": self.__output_names + ["det_kpts"],
                "dynamic_axes": {**self.__dynamic_axes, "det_kpts": {0: "batch"}},
            },
            "Classify": {
                "class_map": {
                    "yolov3": YOLOClassify,
                    "yolov5": YOLOClassify,
                    "yolov8": UltralyticsClassify,
                    "yolo11": UltralyticsClassify,
                    "yolo12": UltralyticsClassify,
                    "ultralytics": UltralyticsClassify,
                },
                "output_names": ["topk"],
                "dynamic_axes": {"images": {0: "batch", 2: "height", 3: "width"}, "topk": {0: "batch"}},
            },
        }
        self.__export_info = {
            'yolov6': "https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800",
            'yolov7': "https://github.com/WongKinYiu/yolov7#export",
            'yolov9': "https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461",
        }
        logger.info("Starting export with Pytorch.")

    def load(self, weights: str, version: str, repo_dir: Optional[str] = None, custom_classes: Optional[Sequence[str]] = None) -> None:
        self.__version = version
        self.__weight_name = Path(weights).stem

        yolo_versions_with_repo = {
            'yolov3': 'ultralytics/yolov3',
            'yolov5': 'ultralytics/yolov5',
        }

        source = 'github' if repo_dir is None else 'local'

        if version in yolo_versions_with_repo:
            repo_dir = yolo_versions_with_repo[version] if repo_dir is None else repo_dir
            self.__model = torch.hub.load(repo_dir, 'custom', path=weights, source=source, _verbose=False)
        elif version in ['yolov8', 'yolov10', 'yolo11', 'yolo12', 'yolo-world', 'yoloe', 'ultralytics']:
            from ultralytics import YOLO

            self.__model = YOLO(model=weights, verbose=False).model

            if custom_classes is not None:
                if version == 'yolo-world':
                    self.__model.set_classes(custom_classes)
                elif version == 'yoloe':
                    self.__model.set_classes(custom_classes, self.__model.get_text_pe(custom_classes))
        else:
            logger.error(
                f"YOLO version '{version}' is unsupported for export with trtyolo CLI tool. "
                "Please provide a valid version, e.g., yolov3, yolov5, yolov8, yolov10, yolo11, yolo12, yolo-world, yoloe, ultralytics."
            )
            if version in self.__export_info:
                logger.warning(
                    f"The official {version} repository supports exporting an ONNX model with the EfficientNMS_TRT plugin. "
                    f"Please refer to {self.__export_info[version]} for instructions on how to export it."
                )
            sys.exit(1)

    def register(self, batch: int, max_boxes: int, iou_thres: float, conf_thres: float) -> None:
        self.__model = deepcopy(self.__model).to(torch.device("cpu"))
        supported = False

        self.__max_boxes = max_boxes
        self.__dynamic = batch <= 0
        self.__batch = 1 if batch <= 0 else batch

        for m in self.__model.modules():
            class_name = m.__class__.__name__
            if class_name in self.__head_config:
                self.__head_name = class_name
                detect_head = self.__head_config[class_name]["class_map"].get(self.__version)
                if detect_head:
                    supported = True
                    if class_name != "Classify":
                        detect_head.dynamic = self.__dynamic
                        detect_head.max_det = max_boxes
                        detect_head.iou_thres = iou_thres
                        detect_head.conf_thres = conf_thres
                    m.__class__ = detect_head
                break

        if not supported:
            logger.error(f"YOLO version '{self.__version}' output head not supported!")
            sys.exit(1)

    def export(self, output: str, imgsz: Union[int, Sequence[int]], opset_version: int, simplify: bool) -> None:
        from ultralytics.utils.checks import check_imgsz
        from ultralytics.utils.torch_utils import TORCH_2_4

        imgsz = check_imgsz(imgsz, stride=self.__model.stride, min_dim=2)
        im = torch.zeros(self.__batch, 3, *imgsz).to(torch.device("cpu"))

        for p in self.__model.parameters():
            p.requires_grad = False
        self.__model.eval()
        self.__model.float()

        preds = self.__model(im)

        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_path = output_path / (self.__weight_name + ".onnx")

        kwargs = {"dynamo": False} if TORCH_2_4 else {}
        torch.onnx.export(
            model=self.__model,
            args=im,
            f=str(onnx_path),
            opset_version=opset_version,
            input_names=['images'],
            output_names=self.__head_config[self.__head_name]["output_names"],
            dynamic_axes=self.__head_config[self.__head_name]["dynamic_axes"] if self.__dynamic else None,
            verbose=False,
            **kwargs,
        )

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        # Update dynamic axes names
        output_shapes = {
            'num_dets': ["batch" if self.__dynamic else self.__batch, 1],
            'det_boxes': ["batch" if self.__dynamic else self.__batch, self.__max_boxes, 4],
            'det_scores': ["batch" if self.__dynamic else self.__batch, self.__max_boxes],
            'det_classes': ["batch" if self.__dynamic else self.__batch, self.__max_boxes],
        }

        if self.__head_name == "OBB":
            output_shapes['det_boxes'] = ["batch" if self.__dynamic else self.__batch, self.__max_boxes, 5]
        elif self.__head_name == "Pose":
            output_shapes['det_kpts'] = [
                "batch" if self.__dynamic else self.__batch,
                self.__max_boxes,
                preds[-1].shape[-2],
                preds[-1].shape[-1],
            ]
        elif self.__head_name == "Segment" or self.__head_name == 'YOLOESegment':
            output_shapes['det_masks'] = [
                "batch" if self.__dynamic else self.__batch,
                self.__max_boxes,
                "height" if self.__dynamic else imgsz[0] // 4,
                "width" if self.__dynamic else imgsz[1] // 4,
            ]
        elif self.__head_name == "Classify":
            output_shapes = {'topk': ["batch" if self.__dynamic else self.__batch, preds[-1].shape[1], 2]}

        for node in onnx_model.graph.output:
            for idx, dim in enumerate(node.type.tensor_type.shape.dim):
                dim.dim_param = str(output_shapes[node.name][idx])

        if simplify:
            try:
                import onnxsim

                logger.success(f"Simplifying ONNX model with onnxsim version {onnxsim.__version__}...")
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, "Simplified ONNX model could not be validated"
            except ImportError:
                logger.warning('onnxsim not found. Please install onnx-simplifier for example: `pip install onnx-simplifier>=0.4.1`.')
            except Exception as e:
                logger.warning(f"Simplifier failure: {e}")

        onnx.save(onnx_model, onnx_path)

        logger.success(f'Export complete, results saved to {output}, visualize at https://netron.app')


def torch_export(
    weights: str,
    output: str,
    version: str,
    imgsz: Optional[Sequence[int]] = [640, 640],
    batch: Optional[int] = 1,
    max_boxes: Optional[int] = 100,
    iou_thres: Optional[float] = 0.45,
    conf_thres: Optional[float] = 0.25,
    opset_version: Optional[int] = 12,
    simplify: Optional[bool] = True,
    repo_dir: Optional[str] = None,
    custom_classes: Optional[Sequence[str]] = None,
) -> None:
    """
    Export YOLO model to ONNX format using Torch.

    Args:
        weights (str): Path to YOLO weights for PyTorch.
        output (str): Directory path to save the exported model.
        version (str): YOLO version, e.g., yolov3, yolov5, yolov8, yolov10, yolo11, yolo12, yolo-world, yoloe, ultralytics.
        imgsz (Optional[Sequence[int]], optional): Inference image size (height, width). Defaults to [640, 640].
        batch (Optional[int], optional): Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.
        max_boxes (Optional[int], optional): Maximum number of detections to output per image. Defaults to 100.
        iou_thres (Optional[float], optional): NMS IoU threshold for post-processing. Defaults to 0.45.
        conf_thres (Optional[float], optional): Confidence threshold for object detection. Defaults to 0.25.
        opset_version (Optional[int], optional): ONNX opset version. Defaults to 12.
        simplify (Optional[bool], optional): Whether to simplify the exported ONNX. Defaults to True.
        repo_dir (Optional[str], optional): Directory containing the local repository (if using torch.hub.load). Defaults to None.
        custom_classes (Optional[Sequence[str]], optional): Custom class names for the YOLO-World and YOLOE. Defaults to None.
    """
    exporter = ModelExporter()
    exporter.load(weights, version, repo_dir, custom_classes)
    exporter.register(batch, max_boxes, iou_thres, conf_thres)
    exporter.export(output, imgsz, opset_version, simplify)


def paddle_export(
    model_dir: str,
    model_filename: str,
    params_filename: str,
    output: str,
    batch: Optional[int] = 1,
    max_boxes: Optional[int] = 100,
    iou_thres: Optional[float] = 0.45,
    conf_thres: Optional[float] = 0.25,
    opset_version: Optional[int] = 12,
    simplify: Optional[bool] = True,
) -> None:
    """
    Export YOLO model to ONNX format using PaddleDetection.

    Args:
        model_dir (str): Path to the directory containing the PaddleDetection PP-YOLOE model.
        model_filename (str): The filename of the PP-YOLOE model.
        params_filename (str): The filename of the PP-YOLOE parameters.
        output (str): Directory path to save the exported model.
        batch (Optional[int], optional): Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.
        max_boxes (Optional[int], optional): Maximum number of detections to output per image. Defaults to 100.
        iou_thres (Optional[float], optional): NMS IoU threshold for post-processing. Defaults to 0.45.
        conf_thres (Optional[float], optional): Confidence threshold for object detection. Defaults to 0.25.
        opset_version (Optional[int], optional): ONNX opset version. Defaults to 12.
        simplify (Optional[bool], optional): Whether to simplify the exported ONNX. Defaults to True.
    """
    from .ppyoloe import PPYOLOEGraphSurgeon

    logger.info("Starting export with PaddlePaddle.")

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_filepath = output_path / (Path(model_filename).stem + ".onnx")

    ppyoloe_gs = PPYOLOEGraphSurgeon(
        model_dir=model_dir,
        onnx_path=str(onnx_filepath),
        model_filename=model_filename,
        params_filename=params_filename,
        opset=opset_version,
        batch_size=1 if batch <= 0 else batch,
        dynamic=batch <= 0,
        simplify=simplify,
    )

    # Register the `EfficientNMS_TRT` into the graph.
    ppyoloe_gs.register_nms(score_thresh=conf_thres, nms_thresh=iou_thres, detections_per_img=max_boxes)

    # Save the exported ONNX models.
    ppyoloe_gs.save(str(onnx_filepath))

    logger.success(f'Export complete, results saved to {output}, visualize at https://netron.app')


def validate_imgsz(ctx: click.Context, param: click.Parameter, value: str) -> Tuple[int, int]:
    """Validate and parse the imgsz parameter."""
    try:
        if ',' in value:
            h, w = map(int, value.split(','))
            return (h, w)
        size = int(value)
        return (size, size)
    except ValueError:
        raise click.BadParameter('Image size must be in format "size" or "height,width" (e.g., 640 or 640,480)')


def validate_names(ctx: click.Context, param: click.Parameter, value: str) -> Optional[List[str]]:
    """Validate and parse the names parameter."""
    if value is None:
        return None
    return [name.strip() for name in value.split(',')]


def validate_export_params(ctx: click.Context, param: click.Parameter, value: str) -> str:
    """Validate the combination of export parameters."""
    version = ctx.params.get('version')
    model_dir = ctx.params.get('model_dir')
    model_filename = ctx.params.get('model_filename')
    params_filename = ctx.params.get('params_filename')
    weights = ctx.params.get('weights')

    if version == 'pp-yoloe':
        if not all([model_dir, model_filename, params_filename]):
            raise click.BadParameter('For PP-YOLOE, --model_dir, --model_filename and --params_filename are required')
    elif version and version != 'pp-yoloe':
        if not weights:
            raise click.BadParameter('For non PP-YOLOE models, --weights is required')

    return value


@click.group()
def trtyolo():
    """Command line tool for exporting models and performing inference with TensorRT-YOLO."""
    pass


@trtyolo.command(
    help="Export YOLO models to ONNX format compatible with TensorRT-YOLO. "
    "Supports YOLOv3, YOLOv5, YOLOv8, YOLOv10, YOLO11, YOLO12, YOLO-World, YOLOE, PP-YOLOE, and PP-YOLOE+."
)
@click.option(
    '-v',
    '--version',
    help='Model version. Options include yolov3, yolov5, yolov8, yolov10, yolo11, yolo12, yolo-world, yoloe, pp-yoloe, ultralytics.',
    type=str,
    required=True,
    callback=validate_export_params,
)
@click.option(
    '-o',
    '--output',
    help='Directory path to save the exported model.',
    type=click.Path(file_okay=False, dir_okay=True, path_type=str),
    required=True,
)
@click.option(
    '-w',
    '--weights',
    help='Path to PyTorch YOLO weights (required for non PP-YOLOE models).',
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=str),
)
@click.option(
    '--model_dir',
    help='Directory path containing the PaddleDetection PP-YOLOE model (required for PP-YOLOE).',
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=str),
)
@click.option('--model_filename', help='Filename of the PaddleDetection PP-YOLOE model (required for PP-YOLOE).', type=str)
@click.option('--params_filename', help='Filename of the PaddleDetection PP-YOLOE parameters (required for PP-YOLOE).', type=str)
@click.option('-b', '--batch', default=1, help='Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.', type=int)
@click.option(
    '--max_boxes',
    default=100,
    help='Maximum number of detections per image (not applicable for classification models). Defaults to 100.',
    type=int,
)
@click.option(
    '--iou_thres',
    default=0.45,
    help='NMS IoU threshold for post-processing (not applicable for classification models). Defaults to 0.45.',
    type=float,
)
@click.option(
    '--conf_thres',
    default=0.25,
    help='Confidence threshold for object detection (not applicable for classification models). Defaults to 0.25.',
    type=float,
)
@click.option(
    '--imgsz',
    default='640',
    help='Image size (single value for square or "height,width"). Defaults to "640" (for non PP-YOLOE models).',
    type=str,
    callback=validate_imgsz,
)
@click.option(
    '-n',
    '--names',
    help='Custom class names for YOLO-World and YOLOE (comma-separated, e.g., "person,car,dog"). Only applicable for YOLO-World and YOLOE models.',
    type=str,
    callback=validate_names,
)
@click.option(
    '--repo_dir',
    help='Directory containing the local repository (if using torch.hub.load). Only applicable for YOLOv3 and YOLOv5 models.',
    type=str,
)
@click.option('--opset', default=12, help='ONNX opset version. Defaults to 12.', type=int)
@click.option('-s', '--simplify', is_flag=True, help='Whether to simplify the exported ONNX model. Defaults to False.')
def export(
    version: str,
    output: str,
    weights: Optional[str],
    model_dir: Optional[str],
    model_filename: Optional[str],
    params_filename: Optional[str],
    imgsz: Optional[Tuple[int, int]],
    names: Optional[List[str]],
    repo_dir: Optional[str],
    batch: Optional[int],
    max_boxes: Optional[int],
    iou_thres: Optional[float],
    conf_thres: Optional[float],
    opset: Optional[int],
    simplify: Optional[bool],
):
    """Export models for TensorRT-YOLO.

    This command allows exporting models for both PaddlePaddle and PyTorch frameworks to be used with TensorRT-YOLO.
    """

    if version == 'pp-yoloe':
        paddle_export(
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch=batch,
            output=output,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset,
            simplify=simplify,
        )
    else:
        torch_export(
            weights=weights,
            output=output,
            version=version,
            imgsz=imgsz,
            batch=batch,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset,
            simplify=simplify,
            repo_dir=repo_dir,
            custom_classes=names,
        )

    logger.success("Export completed successfully!")


if __name__ == '__main__':
    trtyolo()
