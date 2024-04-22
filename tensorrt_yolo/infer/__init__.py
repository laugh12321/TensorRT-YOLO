import sys

from loguru import logger

from .yolo import TRTYOLO
from .batcher import ImageBatcher
from .structs import TensorInfo, DetectInfo
from .general import generate_labels_with_colors, letterbox, scale_boxes, visualize_detections

# For scripts
logger.configure(
    handlers=[dict(sink=sys.stdout, colorize=True, format="<level>[{level.name[0]}]</level> <level>{message}</level>")]
)

__all__ = ['TRTYOLO', 'ImageBatcher', 'TensorInfo', 'DetectInfo', 'generate_labels_with_colors', 'letterbox', 'scale_boxes', 'visualize_detections']
