import sys
from pathlib import Path

import rich_click as click
from loguru import logger

from .export import paddle_export, torch_export
from .infer import TRTYOLO, DetectInfo, ImageBatcher, TensorInfo, generate_labels_with_colors, letterbox, scale_boxes, visualize_detections

__all__ = [
    'trtyolo',
    'TRTYOLO',
    'ImageBatcher',
    'TensorInfo',
    'DetectInfo',
    'generate_labels_with_colors',
    'letterbox',
    'scale_boxes',
    'visualize_detections',
    'torch_export',
    'paddle_export',
]

logger.configure(handlers=[{'sink': sys.stdout, 'colorize': True, 'format': "<level>[{level.name[0]}]</level> <level>{message}</level>"}])


@click.group()
def trtyolo():
    """Command line tool for exporting models and performing inference with TensorRT-YOLO."""
    pass


@trtyolo.command(help="Export models for TensorRT-YOLO. Supports YOLOv5, YOLOv8, YOLOv9, PP-YOLOE and PP-YOLOE+.")
@click.option('--model_dir', help='Path to the directory containing the PaddleDetection PP-YOLOE model.', type=str)
@click.option('--model_filename', help='The filename of the PP-YOLOE model.', type=str)
@click.option('--params_filename', help='The filename of the PP-YOLOE parameters.', type=str)
@click.option('-w', '--weights', help='Path to YOLO weights for PyTorch.', type=str)
@click.option('-v', '--version', help='YOLO version, e.g., yolov5, yolov8, yolov9.', type=str)
@click.option('--imgsz', default=640, help='Inference image size. Defaults to 640.', type=int)
@click.option('--repo_dir', default=None, help='Directory containing the local repository (if using torch.hub.load).', type=str)
@click.option('-o', '--output', help='Directory path to save the exported model.', type=str, required=True)
@click.option('-b', '--batch', default=1, help='Total batch size for the model. Use -1 for dynamic batch size. Defaults to 1.', type=int)
@click.option('--max_boxes', default=100, help='Maximum number of detections to output per image. Defaults to 100.', type=int)
@click.option('--iou_thres', default=0.45, help='NMS IoU threshold for post-processing. Defaults to 0.45.', type=float)
@click.option('--conf_thres', default=0.25, help='Confidence threshold for object detection. Defaults to 0.25.', type=float)
@click.option('--opset_version', default=11, help='ONNX opset version. Defaults to 11.', type=int)
@click.option('-s', '--simplify', is_flag=True, help='Whether to simplify the exported ONNX. Defaults is True.')
def export(
    model_dir,
    model_filename,
    params_filename,
    weights,
    version,
    imgsz,
    repo_dir,
    output,
    batch,
    max_boxes,
    iou_thres,
    conf_thres,
    opset_version,
    simplify,
):
    """Export models for TensorRT-YOLO.

    This command allows exporting models for both PaddlePaddle and PyTorch frameworks to be used with TensorRT-YOLO.
    """
    if model_dir and model_filename and params_filename:
        paddle_export(
            model_dir=model_dir,
            model_filename=model_filename,
            params_filename=params_filename,
            batch=batch,
            output=output,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset_version,
            simplify=simplify,
        )
    elif weights and version:
        torch_export(
            weights=weights,
            output=output,
            version=version,
            imgsz=imgsz,
            batch=batch,
            max_boxes=max_boxes,
            iou_thres=iou_thres,
            conf_thres=conf_thres,
            opset_version=opset_version,
            simplify=simplify,
            repo_dir=repo_dir,
        )
    else:
        logger.error("Please provide correct export parameters.")


@trtyolo.command(help="Perform inference with TensorRT-YOLO.")
@click.option('-e', '--engine', help='Engine file for inference.', type=str, required=True)
@click.option('-i', '--input', help='Input directory or file for inference.', type=str, required=True)
@click.option('-o', '--output', help='Output directory for inference results.', type=str, required=True)
@click.option('-l', '--labels', help='Labels file for inference.', type=str, required=True)
def infer(engine, input, output, labels):
    """Perform inference with TensorRT-YOLO.

    This command performs inference using TensorRT-YOLO with the specified engine file and input source.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    from rich.progress import track

    labels = generate_labels_with_colors(labels)
    model = TRTYOLO(engine)
    model.warmup()

    total_time = 0.0
    total_infers = 0
    total_images = 0
    logger.info(f"Infering data in {input}")
    batcher = ImageBatcher(input_path=input, batch_size=model.batch_size, imgsz=model.imgsz, dtype=model.dtype, dynamic=model.dynamic)

    for batch, images, batch_shape in track(batcher, description="[cyan]Processing batches", total=len(batcher.batches)):
        start_time_ns = time.perf_counter_ns()
        detections = model.infer(batch, batch_shape)
        end_time_ns = time.perf_counter_ns()
        elapsed_time_ms = (end_time_ns - start_time_ns) / 1e6
        total_time += elapsed_time_ms
        total_images += len(images)
        total_infers += 1
        if output:
            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)
            with ThreadPoolExecutor() as executor:
                args_list = [(str(image), str(output_dir / image.name), detections[i], labels) for i, image in enumerate(images)]
                executor.map(visualize_detections, *zip(*args_list))

    average_latency = total_time / total_infers
    average_throughput = total_images / (total_time / 1000)
    logger.success(
        "Benchmark results include time for H2D and D2H memory copies\n"
        f"    CPU Average Latency: {average_latency:.3f} ms\n"
        f"    CPU Average Throughput: {average_throughput:.1f} ips\n"
        "    Finished Inference."
    )
