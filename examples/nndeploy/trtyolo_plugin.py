import json
from typing import List, Optional, Union

import cv2
import nndeploy.base
import nndeploy.dag
import numpy as np
import supervision as sv

from trtyolo import TRTYOLO

# 定义联合类型，表示可能是五个类型中的任意一种
InferRes = Union[sv.Detections, sv.KeyPoints, sv.Classifications]

supported_model_tasks = [
    "detect",
    "segment",
    "classify",
    "pose",
    "obb",
]


class ClassificationAnnotator:
    """
    Ref: https://github.com/roboflow/supervision/commit/9c2aaa0d2ebac2fcb485f625a15059870deae1ee
    Annotate classification results on an image.
    """

    def __init__(
        self,
        color: Union[sv.Color, sv.ColorPalette] = sv.ColorPalette.DEFAULT,
        text_color: sv.Color = sv.Color.BLACK,
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        text_position: sv.Position = sv.Position.TOP_LEFT,
        color_lookup: sv.ColorLookup = sv.ColorLookup.CLASS,
    ):
        """
        Args:
            color (Union[sv.Color, sv.ColorPalette]): The color or color palette to use for
                annotating the text background.
            text_color (sv.Color): The color to use for the text.
            text_scale (float): Font scale for the text.
            text_thickness (int): Thickness of the text characters.
            text_position (sv.Position): Position of the text relative to the image.
                Possible values are defined in the `sv.Position` enum.
            color_lookup (sv.ColorLookup): Strategy for mapping colors to annotations.
                Options are `sv.ColorLookup.INDEX`, `sv.ColorLookup.CLASS`, `sv.ColorLookup.TRACE`.
        """
        self.color: Union[sv.Color, sv.ColorPalette] = color
        self.text_color: sv.Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_position: sv.Position = text_position
        self.color_lookup: sv.ColorLookup = color_lookup
        self.text_padding: int = text_padding

    def annotate(
        self,
        scene: np.ndarray,
        classifications: sv.Classifications,
        labels: List[str] = None,
        custom_color_lookup: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Annotates the given scene with labels based on the provided classifications.
        Args:
            scene (np.ndarray): The image where labels will be drawn.
            classifications (Classifications): Object classifications to annotate.
            labels (List[str]): Optional. Custom labels for each classification.
            custom_color_lookup (Optional[np.ndarray]): Custom color lookup array.
                Allows to override the default color mapping strategy.
        Returns:
            np.ndarray: The annotated image.
        Example:
            ```python
            >>> import supervision as sv
            >>> image = ...
            >>> classifications = sv.Classifications(...)
            >>> classification_annotator = sv.ClassificationAnnotator(text_position=sv.Position.CENTER)
            >>> annotated_frame = classification_annotator.annotate(
            ...     scene=image.copy(),
            ...     classifications=classifications
            ... )
            ```
        """
        font = cv2.FONT_HERSHEY_SIMPLEX

        classification = classifications.get_top_k(k=1)
        classification_idx = classifications.class_id.tolist().index(classification[0][0])

        color = sv.annotators.utils.resolve_color(
            color=self.color,
            detections=classifications,
            detection_idx=classification_idx,
            color_lookup=self.color_lookup if custom_color_lookup is None else custom_color_lookup,
        )

        self.text_color = color

        text = f"{labels[classification_idx]} ({classifications.confidence[classification_idx] * 100:.2f}%)"

        text_wh = cv2.getTextSize(
            text=text,
            fontFace=font,
            fontScale=self.text_scale,
            thickness=self.text_thickness,
        )[0]

        if self.text_position == sv.Position.BOTTOM_LEFT:
            text_x = self.text_padding
            text_y = scene.shape[0] - self.text_padding
        elif self.text_position == sv.Position.BOTTOM_RIGHT:
            text_x = scene.shape[1] - text_wh[0] - self.text_padding
            text_y = scene.shape[0] - self.text_padding
        elif self.text_position == sv.Position.TOP_LEFT:
            text_x = self.text_padding
            text_y = text_wh[1] + self.text_padding
        elif self.text_position == sv.Position.TOP_RIGHT:
            text_x = scene.shape[1] - text_wh[0] - self.text_padding
            text_y = text_wh[1] + self.text_padding
        else:
            raise ValueError(f"Invalid position {self.text_position} for classification annotator.")

        cv2.putText(
            img=scene,
            text=text,
            org=(text_x, text_y),
            fontFace=font,
            fontScale=self.text_scale,
            color=self.text_color.as_rgb(),
            thickness=self.text_thickness,
            lineType=cv2.LINE_AA,
        )
        return scene


class TRTYOLONode(nndeploy.dag.Node):
    def __init__(
        self,
        name,
        inputs: list[nndeploy.dag.Edge] = None,
        outputs: list[nndeploy.dag.Edge] = None,
    ):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.trtyolo.TRTYOLO")
        super().set_desc("TensorRT-YOLO Node")
        self.set_input_type(np.ndarray)
        self.set_output_type(InferRes)

        self.engine_path = ""  # Path to TensorRT engine (*.engine)
        self.model_task = "detect"  # Task in {'detect', 'segment', 'classify', 'pose', 'obb'}
        self.device_id = 0  # GPU device id
        self.swap_rb = True  # Swap R<->B during preprocessing

    def init(self):
        try:
            self.model = TRTYOLO(
                model=self.engine_path,
                task=self.model_task,
                device=self.device_id,
                swap_rb=self.swap_rb,
            )
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"TRTYOLO Node Init Failed: {e}")
            return nndeploy.base.Status.error()

    def run(self):
        try:
            input_numpy = self.get_input_data(0)
            result = self.model.predict(input_numpy)
            self.set_output_data(result, 0)
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"TRTYOLO Node Run Failed: {e}")
            return nndeploy.base.Status.error()

    def serialize(self):
        self.add_io_param("engine_path")
        self.add_required_param("engine_path")
        self.add_dropdown_param("model_task", supported_model_tasks)
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["engine_path"] = self.engine_path
        json_obj["model_task"] = self.model_task
        json_obj["device_id"] = self.device_id
        json_obj["swap_rb"] = self.swap_rb
        return json.dumps(json_obj, ensure_ascii=False, indent=2)

    def deserialize(self, target: str):
        try:
            json_obj = json.loads(target)
            if "engine_path" in json_obj:
                self.engine_path = json_obj["engine_path"]
            if "model_task" in json_obj:
                self.model_task = json_obj["model_task"]
            if "device_id" in json_obj:
                self.device_id = json_obj["device_id"]
            if "swap_rb" in json_obj:
                self.swap_rb = json_obj["swap_rb"]
            return super().deserialize(target)
        except Exception as e:
            print(f"TRTYOLO Node Deserialize Failed: {e}")
            return nndeploy.base.Status.error()


class VisualizerNode(nndeploy.dag.Node):
    def __init__(
        self,
        name,
        inputs: list[nndeploy.dag.Edge] = None,
        outputs: list[nndeploy.dag.Edge] = None,
    ):
        super().__init__(name, inputs, outputs)
        super().set_key("nndeploy.trtyolo.Visualizer")
        super().set_desc("TensorRT-YOLO Visualizer Node")
        self.set_input_type(np.ndarray)
        self.set_input_type(InferRes)
        self.set_output_type(np.ndarray)

        self.labels_file = ""  # Path to label file (*.txt)
        self.model_task = "detect"  # Task in {'detect', 'segment', 'classify', 'pose', 'obb'}

        self.classification_annotator: Optional[ClassificationAnnotator] = None
        self.box_annotator: Optional[sv.BoxAnnotator] = None
        self.mask_annotator: Optional[sv.MaskAnnotator] = None
        self.label_annotator: Optional[sv.LabelAnnotator] = None
        self.oriented_box_annotator: Optional[sv.OrientedBoxAnnotator] = None
        self.vertex_annotator: Optional[sv.VertexAnnotator] = None

    def init(self):
        try:
            self.class_name = [line.strip() for line in open(self.labels_file, "r")]

            if self.model_task == "classify":
                self.classification_annotator = ClassificationAnnotator()
            elif self.model_task == "obb":
                self.oriented_box_annotator = sv.OrientedBoxAnnotator()
                self.label_annotator = sv.LabelAnnotator()
            elif self.model_task == "pose":
                self.box_annotator = sv.BoxAnnotator()
                self.vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=3)
                self.label_annotator = sv.LabelAnnotator()
            elif self.model_task == "segment":
                self.box_annotator = sv.BoxAnnotator()
                self.mask_annotator = sv.MaskAnnotator()
                self.label_annotator = sv.LabelAnnotator()
            else:
                self.box_annotator = sv.BoxAnnotator()
                self.label_annotator = sv.LabelAnnotator()

            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Visualizer Node Init Failed: {e}")
            return nndeploy.base.Status.error()

    def run(self):
        try:
            vis_image = self.get_input_data(0)
            infer_res = self.get_input_data(1)

            if infer_res.class_id is None:
                self.set_output_data(vis_image, 0)
                return nndeploy.base.Status.ok()

            if self.model_task == "classify":
                labels = [self.class_name[int(cls)] for cls in infer_res.class_id]
                vis_image = self.classification_annotator.annotate(scene=vis_image, classifications=infer_res, labels=labels)
            elif self.model_task == "obb":
                labels = [f"{self.class_name[int(cls)]} {conf:.2f}" for cls, conf in zip(infer_res.class_id, infer_res.confidence)]
                vis_image = self.oriented_box_annotator.annotate(scene=vis_image, detections=infer_res)
                vis_image = self.label_annotator.annotate(scene=vis_image, detections=infer_res, labels=labels)
            elif self.model_task == "pose":
                labels = [self.class_name[int(cls)] for cls in infer_res.class_id]
                detections = infer_res.as_detections()
                vis_image = self.box_annotator.annotate(scene=vis_image, detections=detections)
                vis_image = self.vertex_annotator.annotate(scene=vis_image, key_points=infer_res)
                vis_image = self.label_annotator.annotate(scene=vis_image, detections=detections, labels=labels)
            elif self.model_task == "segment":
                labels = [f"{self.class_name[int(cls)]} {conf:.2f}" for cls, conf in zip(infer_res.class_id, infer_res.confidence)]
                vis_image = self.box_annotator.annotate(scene=vis_image, detections=infer_res)
                vis_image = self.mask_annotator.annotate(scene=vis_image, detections=infer_res)
                vis_image = self.label_annotator.annotate(scene=vis_image, detections=infer_res, labels=labels)
            else:
                labels = [f"{self.class_name[int(cls)]} {conf:.2f}" for cls, conf in zip(infer_res.class_id, infer_res.confidence)]
                vis_image = self.box_annotator.annotate(scene=vis_image, detections=infer_res)
                vis_image = self.label_annotator.annotate(scene=vis_image, detections=infer_res, labels=labels)

            self.set_output_data(vis_image, 0)
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Visualizer Node Run Failed: {e}")
            return nndeploy.base.Status.error()

    def serialize(self):
        self.add_io_param("labels_file")
        self.add_required_param("labels_file")
        self.add_dropdown_param("model_task", supported_model_tasks)
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["labels_file"] = self.labels_file
        json_obj["model_task"] = self.model_task
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        try:
            json_obj = json.loads(target)
            if "labels_file" in json_obj:
                self.labels_file = json_obj["labels_file"]
            if "model_task" in json_obj:
                self.model_task = json_obj["model_task"]
            return super().deserialize(target)
        except Exception as e:
            print(f"Visualizer Node Deserialize Failed: {e}")
            return nndeploy.base.Status.error()


class TRTYOLONodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()

    def create_node(
        self,
        name: str,
        inputs: list[nndeploy.dag.Edge],
        outputs: list[nndeploy.dag.Edge],
    ):
        self.node = TRTYOLONode(name, inputs, outputs)
        return self.node


class VisualizerNodeCreator(nndeploy.dag.NodeCreator):
    def __init__(self):
        super().__init__()

    def create_node(
        self,
        name: str,
        inputs: list[nndeploy.dag.Edge],
        outputs: list[nndeploy.dag.Edge],
    ):
        self.node = VisualizerNode(name, inputs, outputs)
        return self.node


trtyolo_node_creator = TRTYOLONodeCreator()
visualizer_node_creator = VisualizerNodeCreator()

nndeploy.dag.register_node("nndeploy.trtyolo.TRTYOLO", trtyolo_node_creator)
nndeploy.dag.register_node("nndeploy.trtyolo.Visualizer", visualizer_node_creator)
