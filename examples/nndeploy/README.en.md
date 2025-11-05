[简体中文](README.md) | English

# Using TensorRT-YOLO in nndeploy Workflow

This example briefly introduces how to use the [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) toolkit in nndeploy's workflow to implement multi-class computer vision tasks, including **object detection, instance segmentation, image classification, pose estimation, and oriented object detection**.

<div align="center">
    <table>
    <tr>
        <td><img width="100%" src="../../assets/nndeploy_detect.png"></td>
        <td><img width="100%" src="../../assets/nndeploy_obb.png"></td>
    </tr>
    </table>
</div>

## Environment Preparation

Refer to the [TensorRT-YOLO Quick Compilation and Installation](https://github.com/laugh12321/TensorRT-YOLO/blob/main/docs/en/build_and_install.md) document to perform compilation and Python library packaging processes.

> **Note**: Ensure that the `BUILD_PYTHON` option is enabled during compilation, then follow the document steps to complete the packaging and installation of the Python library.

## 2. Model Conversion

Refer to the methods in [TensorRT-YOLO Model Export](https://github.com/laugh12321/TensorRT-YOLO/blob/main/docs/en/model_export.md). First, export the model to ONNX format using the command-line tool (CLI) of the model's Python library, then convert the ONNX model to a TensorRT engine.

## 3. Adding Custom Nodes

<div align="center">
    <p>
        <img width="100%" src="../../assets/nndeploy_plugin.png">
    </p>
</div>

1. In the nndeploy workflow panel, click the `Nodes` column in the left toolbar.

2. Click the `+` button to import the downloaded [`trtyolo_plugin.py`](trtyolo_plugin.py) file.

3. After successful import, refresh the page, and you can view the custom node in the left `Nodes` column.

## 4. Node Parameter Description

### nndeploy.trtyolo.TRTYOLO

<div align="center">

| Parameter Name | Description |
|----------------|-------------|
| engine_path    | Path to the TensorRT engine file |
| model_task     | Type of task performed by the model, optional values: detect, segment, classify, pose, obb |
| device_id      | GPU device ID used for inference |
| swap_rb        | Whether to swap the R and B channels of the image during preprocessing |

</div>

### nndeploy.trtyolo.Visualizer

<div align="center">

| Parameter Name | Description |
|----------------|-------------|
| labels_file    | Path to the label file (*.txt), where each line corresponds to a class name |

</div>
