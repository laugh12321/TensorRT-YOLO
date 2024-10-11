import sys
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

try:
    import torch
except ImportError:
    logger.error('Pytorch not found, plaese install Pytorch.' 'for example: `pip install torch`.')
    sys.exit(1)

try:
    from ultralytics import YOLO
    from ultralytics.utils.checks import check_imgsz
except ImportError:
    logger.error('Ultralytics not found, plaese install Ultralytics.' 'for example: `pip install ultralytics`.')
    sys.exit(1)

from .head import UltralyticsDetect, UltralyticsOBB, YOLODetect, v10Detect
from .ppyoloe import PPYOLOEGraphSurgeon

__all__ = ['torch_export', 'paddle_export']

# Filter warnings
warnings.filterwarnings("ignore")

# For scripts
logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])

HEADS = {
    "Detect": {"yolov3": YOLODetect, "yolov5": YOLODetect, "yolov8": UltralyticsDetect, "yolo11": UltralyticsDetect, "ultralytics": UltralyticsDetect},
    "v10Detect": {"yolov10": v10Detect, "ultralytics": v10Detect},
    "OBB": {"yolov8": UltralyticsOBB, "yolo11": UltralyticsOBB, "ultralytics": UltralyticsOBB},
}

DEFAULT_OUTPUT_NAMES = ["num_dets", "det_boxes", "det_scores", "det_classes"]
DEFAULT_DYNAMIC_AXES = {
    "images": {0: "batch", 2: "height", 3: "width"},
    "num_dets": {0: "batch"},
    "det_boxes": {0: "batch"},
    "det_scores": {0: "batch"},
    "det_classes": {0: "batch"},
}

OUTPUT_NAMES = {
    "Detect": DEFAULT_OUTPUT_NAMES,
    "v10Detect": DEFAULT_OUTPUT_NAMES,
    "OBB": DEFAULT_OUTPUT_NAMES,
}

DYNAMIC_AXES = {
    "Detect": DEFAULT_DYNAMIC_AXES,
    "v10Detect": DEFAULT_DYNAMIC_AXES,
    "OBB": DEFAULT_DYNAMIC_AXES,
}

YOLO_EXPORT_INFO = {
    'yolov6': "https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX#tensorrt-backend-tensorrt-version-800",
    'yolov7': "https://github.com/WongKinYiu/yolov7#export",
    'yolov9': "https://github.com/WongKinYiu/yolov9/issues/130#issue-2162045461",
}


def load_model(version: str, weights: str, repo_dir: Optional[str] = None) -> Optional[torch.nn.Module]:
    """
    Load YOLO model based on version and weights.

    Args:
        version (str): YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, ultralytics.
        weights (str): Path to YOLO weights for PyTorch.
        repo_dir (Optional[str], optional): Directory containing the local repository (if using torch.hub.load). Defaults to None.

    Returns:
        torch.nn.Module: Loaded YOLO model or None if the version is not supported.
    """
    yolo_versions_with_repo = {
        'yolov3': 'ultralytics/yolov3',
        'yolov5': 'ultralytics/yolov5',
    }

    source = 'github' if repo_dir is None else 'local'

    if version in yolo_versions_with_repo:
        repo_dir = yolo_versions_with_repo[version] if repo_dir is None else repo_dir
        return torch.hub.load(repo_dir, 'custom', path=weights, source=source, verbose=False)
    elif version in ['yolov8', 'yolov10', 'yolo11', 'ultralytics']:
        return YOLO(model=weights, verbose=False).model
    elif version in YOLO_EXPORT_INFO:
        logger.warning(
            f"The official {version} repository supports exporting an ONNX model with the EfficientNMS_TRT plugin. "
            f"Please refer to {YOLO_EXPORT_INFO[version]} for instructions on how to export it."
        )
        return None
    else:
        logger.error(f"YOLO version '{version}' not supported!")
        return None


def update_model(
    model: torch.nn.Module, version: str, dynamic: bool, max_boxes: int, iou_thres: float, conf_thres: float
) -> Tuple[Optional[torch.nn.Module], str]:
    """
    Update YOLO model with dynamic settings.

    Args:
        model (torch.nn.Module): YOLO model to be updated.
        version (str): YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, ultralytics.
        dynamic (bool): Whether to use dynamic settings.
        max_boxes (int): Maximum number of detections to output per image.
        iou_thres (float): NMS IoU threshold for post-processing.
        conf_thres (float): Confidence threshold for object detection.

    Returns:
        Tuple[Optional[torch.nn.Module], str]:
            - Updated YOLO model or None if the version is not supported.
            - The name of the detection head.
    """
    model = deepcopy(model).to(torch.device("cpu"))
    supported = False
    head_name = ""

    for m in model.modules():
        class_name = m.__class__.__name__
        if class_name in HEADS:
            head_name = class_name
            detect_head = HEADS[class_name].get(version)
            if detect_head:
                supported = True
                detect_head.dynamic = dynamic
                detect_head.max_det = max_boxes
                detect_head.iou_thres = iou_thres
                detect_head.conf_thres = conf_thres
                m.__class__ = detect_head
            break

    if not supported:
        logger.error(f"YOLO version '{version}' detect head not supported!")
        return None, head_name

    return model, head_name


def torch_export(
    weights: str,
    output: str,
    version: str,
    imgsz: Optional[int] = 640,
    batch: Optional[int] = 1,
    max_boxes: Optional[int] = 100,
    iou_thres: Optional[float] = 0.45,
    conf_thres: Optional[float] = 0.25,
    opset_version: Optional[int] = 11,
    simplify: Optional[bool] = True,
    repo_dir: Optional[str] = None,
) -> None:
    """
    Export YOLO model to ONNX format using Torch.

    Args:
        weights (str): Path to YOLO weights for PyTorch.
        output (str): Directory path to save the exported model.
        version (str): YOLO version, e.g., yolov3, yolov5, yolov6, yolov7, yolov8, yolov9, yolov10, yolo11, ultralytics.
        imgsz (Optional[int], optional): Inference image size. Defaults to 640.
        batch (Optional[int], optional): Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.
        max_boxes (Optional[int], optional): Maximum number of detections to output per image. Defaults to 100.
        iou_thres (Optional[float], optional): NMS IoU threshold for post-processing. Defaults to 0.45.
        conf_thres (Optional[float], optional): Confidence threshold for object detection. Defaults to 0.25.
        opset_version (Optional[int], optional): ONNX opset version. Defaults to 11.
        simplify (Optional[bool], optional): Whether to simplify the exported ONNX. Defaults to True.
        repo_dir (Optional[str], optional): Directory containing the local repository (if using torch.hub.load). Defaults to None.
    """
    logger.info("Starting export with Pytorch.")
    model = load_model(version, weights, repo_dir)
    if model is None:
        return

    dynamic = batch <= 0
    batch = 1 if dynamic else batch
    model, head_name = update_model(model, version, dynamic, max_boxes, iou_thres, conf_thres)
    if model is None:
        return

    imgsz = check_imgsz(imgsz, stride=model.stride, min_dim=2)

    im = torch.zeros(batch, 3, *imgsz).to(torch.device("cpu"))

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.float()
    for _ in range(2):  # Warm-up run
        model(im)

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_filepath = output_path / (Path(weights).stem + ".onnx")

    torch.onnx.export(
        model=model,
        args=im,
        f=str(onnx_filepath),
        opset_version=opset_version,
        input_names=['images'],
        output_names=OUTPUT_NAMES[head_name],
        dynamic_axes=DYNAMIC_AXES[head_name] if dynamic else None,
    )

    try:
        import onnx
    except ImportError:
        logger.error('onnx not found, plaese install onnx.' 'for example: `pip install onnx>=1.12.0`.')
        sys.exit(1)

    model_onnx = onnx.load(onnx_filepath)
    onnx.checker.check_model(model_onnx)

    # Update dynamic axes names
    shapes = {
        'num_dets': ["batch" if dynamic else batch, 1],
        'det_boxes': ["batch" if dynamic else batch, max_boxes, 4],
        'det_scores': ["batch" if dynamic else batch, max_boxes],
        'det_classes': ["batch" if dynamic else batch, max_boxes],
    }
    if head_name == "OBB":
        shapes['det_boxes'] = ["batch" if dynamic else batch, max_boxes, 5]

    for node in model_onnx.graph.output:
        for idx, dim in enumerate(node.type.tensor_type.shape.dim):
            dim.dim_param = str(shapes[node.name][idx])

    if simplify:
        try:
            import onnxsim

            logger.success(f"Simplifying with onnxsim {onnxsim.__version__}...")
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, "Simplified ONNX model could not be validated"
        except Exception as e:
            logger.warning(f"Simplifier failure: {e}")

    onnx.save(model_onnx, onnx_filepath)

    logger.success(f'Export complete, results saved to {output}, visualize at https://netron.app')


def paddle_export(
    model_dir: str,
    model_filename: str,
    params_filename: str,
    output: str,
    batch: Optional[int] = 1,
    max_boxes: Optional[int] = 100,
    iou_thres: Optional[float] = 0.45,
    conf_thres: Optional[float] = 0.25,
    opset_version: Optional[int] = 11,
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
        opset_version (Optional[int], optional): ONNX opset version. Defaults to 11.
        simplify (Optional[bool], optional): Whether to simplify the exported ONNX. Defaults to True.
    """
    logger.info("Staring export with PaddlePaddle.")
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
