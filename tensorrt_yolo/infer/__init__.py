import sys

from loguru import logger

from .batcher import ImageBatcher
from .general import generate_labels_with_colors, letterbox, scale_boxes, visualize_detections
from .structs import DetectInfo, TensorInfo
from .yolo import TRTYOLO

# For scripts
logger.configure(handlers=[{"sink": sys.stdout, "colorize": True, "format": "<level>[{level.name[0]}]</level> <level>{message}</level>"}])

__all__ = [
    'TRTYOLO',
    'ImageBatcher',
    'TensorInfo',
    'DetectInfo',
    'generate_labels_with_colors',
    'letterbox',
    'scale_boxes',
    'visualize_detections',
]
