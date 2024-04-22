import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional

import onnx
import torch
from loguru import logger

from .ppyoloe import PPYOLOEGraphSurgeon
from .head import Detectv5, Detectv8, Detectv9, DDetect, DualDetect, DualDDetect
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

__all__ = ['torch_export', 'paddle_export']

# Filter warnings
warnings.filterwarnings("ignore")

# For scripts
logger.configure(
    handlers=[dict(sink=sys.stdout, colorize=True, format="<level>[{level.name[0]}]</level> <level>{message}</level>")]
)

DETECT_HEADS = {
    "Detect": {"yolov5": Detectv5, "yolov8": Detectv8, "yolov9": Detectv9},
    "DDetect": {"yolov9": DDetect},
    "DualDetect": {"yolov9": DualDetect},
    "DualDDetect": {"yolov9": DualDDetect}
}

OUTPUT_NAMES = ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']

DYNAMIC = {
    "images": {0: "batch", 2: "height", 3: "width"},
    "num_detections": {0: "batch"},
    "detection_boxes": {0: "batch"},
    "detection_scores": {0: "batch"},
    "detection_classes": {0: "batch"},
}


def load_model(version: str, weights: str, repo_dir: Optional[str] = None) -> torch.nn.Module:
    """
    Load YOLO model based on version and weights.

    Args:
        version (str): YOLO version.
        weights (str): Path to the model weights.
        repo_dir (Optional[str], optional): Directory of the YOLO repository. Defaults to None.

    Returns:
        torch.nn.Module: Loaded YOLO model.
    """
    source = 'github' if repo_dir is None else 'local'
    if version == 'yolov5':
        repo_dir = 'ultralytics/yolov5' if repo_dir is None else repo_dir
        return torch.hub.load(repo_dir, 'custom', path=weights, source=source, verbose=False)
    elif version == 'yolov8':
        return YOLO(model=weights, verbose=False).model
    elif version == 'yolov9':
        repo_dir = 'WongKinYiu/yolov9' if repo_dir is None else repo_dir
        return torch.hub.load(repo_dir, 'custom', path=weights, source=source, verbose=False)
    else:
        logger.error(f"YOLO version '{version}' not supported!")
        return None


def update_model(model: torch.nn.Module, version: str, dynamic: bool, max_boxes: int, iou_thres: float, conf_thres: float) -> torch.nn.Module:
    """
    Update YOLO model with dynamic settings.

    Args:
        model (torch.nn.Module): YOLO model to be updated.
        version (str): YOLO version.
        dynamic (bool): Whether to use dynamic settings.
        max_boxes (int): Maximum number of boxes.
        iou_thres (float): IoU threshold.
        conf_thres (float): Confidence threshold.

    Returns:
        torch.nn.Module: Updated YOLO model.
    """
    supported = False
    model = deepcopy(model).to(torch.device("cpu"))
    for m in model.modules():
        class_name = m.__class__.__name__
        if class_name in DETECT_HEADS:
            detect_head = DETECT_HEADS[class_name].get(version, None)
            if detect_head is None:
                break

            supported = True

            if version == 'yolov9':
                detect_head.export = True
            detect_head.export = True
            detect_head.dynamic = dynamic
            detect_head.max_det = max_boxes
            detect_head.iou_thres = iou_thres
            detect_head.conf_thres = conf_thres
            setattr(m, '__class__', detect_head)

    if not supported:
        logger.error(f"YOLO version '{version}' detect head not supported!")
        return None

    return model


def torch_export(weights: str, output: str, version: str, imgsz: Optional[int] = 640, batch: Optional[int] = 1, max_boxes: Optional[int] = 100,
                 iou_thres: Optional[float] = 0.45, conf_thres: Optional[float] = 0.25, opset_version: Optional[int] = 11, simplify: Optional[bool] = True,
                 repo_dir: Optional[str] = None) -> None:
    """
    Export YOLO model to ONNX format using Torch.

    Args:
        weights (str): Path to the model weights.
        output (str): Output directory for the ONNX file.
        version (str): YOLO version.
        imgsz (int, optional): Input image size. Defaults to 640.
        batch (int, optional): Batch size. Defaults to 1.
        max_boxes (int, optional): Maximum number of boxes. Defaults to 100.
        iou_thres (float, optional): IoU threshold. Defaults to 0.45.
        conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
        opset_version (int, optional): ONNX opset version. Defaults to 11.
        simplify (bool, optional): Whether to simplify the ONNX model. Defaults to True.
        repo_dir (Optional[str], optional): Directory of the YOLO repository. Defaults to None.
    """
    model = load_model(version, weights, repo_dir)
    if model is None:
        return

    dynamic = batch <= 0
    batch = 1 if batch <= 0 else batch
    model = update_model(model, version, dynamic, max_boxes, iou_thres, conf_thres)
    if model is None:
        return

    imgsz = check_imgsz(imgsz, stride=model.stride, min_dim=2)

    im = torch.zeros(batch, 3, *imgsz).to(torch.device("cpu"))

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    for _ in range(2):
        model(im)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    f = str(Path(output, Path(weights).stem).with_suffix(".onnx"))

    torch.onnx.export(
        model=model,
        args=im,
        f=f,
        opset_version=opset_version,
        input_names=['images'],
        output_names=OUTPUT_NAMES,
        dynamic_axes=DYNAMIC if dynamic else None
    )

    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)

    shapes = {
        'num_detections': ["batch" if dynamic else batch, 1],
        'detection_boxes': ["batch" if dynamic else batch, max_boxes, 4],
        'detection_scores': ["batch" if dynamic else batch, max_boxes],
        'detection_classes': ["batch" if dynamic else batch, max_boxes]
    }
    for node in model_onnx.graph.output:
        for idx, dim in enumerate(node.type.tensor_type.shape.dim):
            dim.dim_param = str(shapes[node.name][idx])

    if simplify:
        try:
            import onnxsim
            logger.info(f"simplifying with onnxsim {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
        except Exception as e:
            logger.warning(f"simplifier failure: {e}")

    onnx.save(model_onnx, f)

    logger.info(f'Export complete, Results saved to {output}, Visualize at https://netron.app')


def paddle_export(model_dir: str, model_filename: str, params_filename: str, output: str, batch: Optional[int] = 1,
                  max_boxes: Optional[int] = 100,
                  iou_thres: Optional[float] = 0.45, conf_thres: Optional[float] = 0.25, opset_version: Optional[int] = 11, simplify: Optional[bool] = True) -> None:
    """
    Export YOLO model to ONNX format using PaddleDetection.

    Args:
        model_dir (str): Path of directory saved PaddleDetection PP-YOLOE model.
        model_filename (str): The PP-YOLOE model file name.
        params_filename (str): The PP-YOLOE parameters file name.
        output (str): Output directory for the ONNX file.
        batch (int, optional): Batch size. Defaults to 1.
        max_boxes (int, optional): Maximum number of boxes. Defaults to 100.
        iou_thres (float, optional): IoU threshold. Defaults to 0.45.
        conf_thres (float, optional): Confidence threshold. Defaults to 0.25.
        opset_version (int, optional): ONNX opset version. Defaults to 11.
        simplify (bool, optional): Whether to simplify the ONNX model. Defaults to True.
    """

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    f = str(Path(output, Path(model_filename).stem).with_suffix(".onnx"))

    ppyoloe_gs = PPYOLOEGraphSurgeon(
        model_dir=model_dir,
        onnx_path=f,
        model_filename=model_filename,
        params_filename=params_filename,
        opset=opset_version,
        batch_size=1 if batch <= 0 else batch,
        dynamic=batch <= 0,
        simplify=simplify
    )

    # Register the `EfficientNMS_TRT` into the graph.
    ppyoloe_gs.register_nms(
        score_thresh=conf_thres,
        nms_thresh=iou_thres,
        detections_per_img=max_boxes
    )

    # Save the exported ONNX models.
    ppyoloe_gs.save(f)

    logger.info(f'Export complete, Results saved to {output}, Visualize at https://netron.app')
