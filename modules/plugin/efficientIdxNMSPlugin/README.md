# Efficient NMS Plugin With Indices

#### Table of Contents
- [Description](#description)
- [Structure](#structure)
  * [Inputs](#inputs)
  * [Dynamic Shape Support](#dynamic-shape-support)
  * [Box Coding Type](#box-coding-type)
  * [Outputs](#outputs)
  * [Parameters](#parameters)
- [Documentation](#documentation)

## Description

The `EfficientIdxNMS_TRT` plugin implements an efficient algorithm to perform Non Maximum Suppression for object detection networks.

This plugin is primarily intended for using with EfficientDet on TensorRT, as this network is particularly sensitive to the latencies introduced by slower NMS implementations. However, the plugin is generic enough that it will work correctly for other detections architectures, such as SSD or FasterRCNN.

This plugin is modified from TensorRT [efficientNMSPlugin](https://github.com/NVIDIA/TensorRT/tree/release/10.2/plugin/efficientNMSPlugin) but with an additional output layer that returns the indices of the detections detection_indices.

## Structure

### Inputs

The plugin has two modes of operation, depending on the given input data. The plugin will automatically detect which mode to operate as, depending on the number of inputs it receives, as follows:

1. **Standard NMS Mode:** Only two input tensors are given, (i) the bounding box coordinates and (ii) the corresponding classification scores for each box.

2. **Fused Box Decoder Mode:** Three input tensors are given, (i) the raw localization predictions for each box originating directly from the localization head of the network, (ii) the corresponding classification scores originating from the classification head of the network, and (iii) the default anchor box coordinates usually hardcoded as constant tensors in the network.

Most object detection networks work by generating raw predictions from a "localization head" which adjust the coordinates of standard non-learned anchor coordinates to produce a tighter fitting bounding box. This process is called "box decoding", and it usually involves a large number of element-wise operations to transform the anchors to final box coordinates. As this can involve exponential operations on a large number of anchors, it can be computationally expensive, so this plugin gives the option of fusing the box decoder within the NMS operation which can be done in a far more efficient manner, resulting in lower latency for the network.

#### Boxes Input
> **Input Shape:** `[batch_size, number_boxes, 4]` or `[batch_size, number_boxes, number_classes, 4]`
>
> **Data Type:** `float32` or `float16`

The boxes input can have 3 dimensions in case a single box prediction is produced for all classes (such as in EfficientDet or SSD), or 4 dimensions when separate box predictions are generated for each class (such as in FasterRCNN), in which case `number_classes` >= 1 and must match the number of classes in the scores input. The final dimension represents the four coordinates that define the bounding box prediction.

For *Standard NMS* mode, this tensor should contain the final box coordinates for each predicted detection. For *Fused Box Decoder* mode, this tensor should have the raw localization predictions. In either case, this data is given as `4` coordinates which makes up the final shape dimension.

#### Scores Input
> **Input Shape:** `[batch_size, number_boxes, number_classes]`
>
> **Data Type:** `float32` or `float16`

The scores input has `number_classes` elements with the predicted scores for each candidate class for each of the `number_boxes` anchor boxes.

Usually, the score values will have passed through a sigmoid activation function before reaching the NMS operation. However, as an optimization, the pre-sigmoid raw scores can also be provided to the NMS plugin to reduce overall network latency. If raw scores are given, enable the `score_activation` parameter so they are processed accordingly.

#### Anchors Input (Optional)
> **Input Shape:** `[1, number_boxes, 4]` or `[batch_size, number_boxes, 4]`
>
> **Data Type:** `float32` or `float16`

Only used in *Fused Box Decoder* mode. It is much more efficient to perform the box decoding within this plugin. In this case, the boxes input will be treated as the raw localization head box corrections, and this third input should contain the default anchor/prior box coordinates.

When used, the input must have 3 dimensions, where the first one may be either `1` in case anchors are constant for all images in a batch, or `batch_size` in case each image has different anchors -- such as in the box refinement NMS of FasterRCNN's second stage.

### Dynamic Shape Support

Most input shape dimensions, namely `batch_size`, `number_boxes`, and `number_classes`, for all inputs can be defined dynamically at runtime if the TensorRT engine is built with dynamic input shapes. However, once defined, these dimensions must match across all tensors that use them (e.g. the same `number_boxes` dimension must be given for both boxes and scores, etc.)

### Box Coding Type
Different object detection networks represent their box coordinate system differently. The two types supported by this plugin are:

1. **BoxCorners:** The four coordinates represent `[x1, y1, x2, y2]` values, where each x,y pair defines the top-left and bottom-right corners of a bounding box.
2. **BoxCenterSize:** The four coordinates represent `[x, y, w, h]` values, where the x,y pair define the box center location, and the w,h pair define its width and height.

Note that for NMS purposes, horizontal and vertical coordinates are fully interchangeable. TensorFlow-trained networks, for example, often uses vertical-first coordinates such as `[y1, x1, y2, x2]`, but this coordinate system will work equally well under the BoxCorner coding. Similarly, `[y, x, h, w]` will be properly covered by the BoxCornerSize coding.

In *Fused Box Decoder* mode, the boxes and anchor tensors should both use the same coding.

### Outputs

The following five output tensors are generated:

- **num_detections:**
  This is a `[batch_size, 1]` tensor of data type `int32`. The last dimension is a scalar indicating the number of valid detections per batch image. It can be less than `max_output_boxes`. Only the top `num_detections[i]` entries in `nms_boxes[i]`, `nms_scores[i]` and `nms_classes[i]` are valid.

- **detection_boxes:**
  This is a `[batch_size, max_output_boxes, 4]` tensor of data type `float32` or `float16`, containing the coordinates of non-max suppressed boxes. The output coordinates will always be in BoxCorner format, regardless of the input code type.

- **detection_scores:**
  This is a `[batch_size, max_output_boxes]` tensor of data type `float32` or `float16`, containing the scores for the boxes.

- **detection_classes:**
  This is a `[batch_size, max_output_boxes]` tensor of data type `int32`, containing the classes for the boxes.

- **detection_indices:**
  This is a `[batch_size, max_output_boxes]` tensor of data type `int32`, containing the indices for the boxes.

### Parameters

| Type     | Parameter                | Description
|----------|--------------------------|--------------------------------------------------------
|`float`   |`score_threshold` *       |The scalar threshold for score (low scoring boxes are removed).
|`float`   |`iou_threshold`           |The scalar threshold for IOU (additional boxes that have high IOU overlap with previously selected boxes are removed).
|`int`     |`max_output_boxes`        |The maximum number of detections to output per image.
|`int`     |`background_class`        |The label ID for the background class. If there is no background class, set it to `-1`.
|`bool`    |`score_activation` *      |Set to true to apply sigmoid activation to the confidence scores during NMS operation.
|`bool`    |`class_agnostic`          |Set to true to do class-independent NMS; otherwise, boxes of different classes would be considered separately during NMS.
|`int`     |`box_coding`              |Coding type used for boxes (and anchors if applicable), 0 = BoxCorner, 1 = BoxCenterSize.

Parameters marked with a `*` have a non-negligible effect on runtime latency. See the [Performance Tuning](#performance-tuning) section below for more details on how to set them optimally.

# Documentation
- [NMS algorithm](https://www.coursera.org/lecture/convolutional-neural-networks/non-max-suppression-dvrjH)
- [NonMaxSuppression ONNX Op](https://github.com/onnx/onnx/blob/master/docs/Operators.md#NonMaxSuppression)
