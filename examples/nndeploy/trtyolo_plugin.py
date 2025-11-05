import json
from typing import Union

import nndeploy.base
import nndeploy.dag
import numpy as np

from tensorrt_yolo.infer import (ClassifyModel, ClassifyRes, DetectModel,
                                 DetectRes, InferOption, OBBModel, OBBRes,
                                 PoseModel, PoseRes, SegmentModel, SegmentRes,
                                 generate_labels, visualize)

# 定义联合类型，表示可能是五个类型中的任意一种
InferRes = Union[ClassifyRes, DetectRes, OBBRes, SegmentRes, PoseRes]

supported_model_tasks = [
    "detect",
    "segment",
    "classify",
    "pose",
    "obb",
]


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
            option = InferOption()
            option.set_device_id(self.device_id)
            if self.swap_rb:
                option.enable_swap_rb()

            self.model = (
                DetectModel(self.engine_path, option)
                if self.model_task == "detect"
                else (
                    OBBModel(self.engine_path, option)
                    if self.model_task == "obb"
                    else (
                        SegmentModel(self.engine_path, option)
                        if self.model_task == "segment"
                        else (PoseModel(self.engine_path, option) if self.model_task == "pose" else ClassifyModel(self.engine_path, option))
                    )
                )
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

    def init(self):
        try:
            self.labels = generate_labels(self.labels_file)
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Visualizer Node Init Failed: {e}")
            return nndeploy.base.Status.error()

    def run(self):
        try:
            image = self.get_input_data(0)
            infer_res = self.get_input_data(1)
            vis_image = visualize(image, infer_res, self.labels)
            self.set_output_data(vis_image, 0)
            return nndeploy.base.Status.ok()
        except Exception as e:
            print(f"Visualizer Node Run Failed: {e}")
            return nndeploy.base.Status.error()

    def serialize(self):
        self.add_io_param("labels_file")
        self.add_required_param("labels_file")
        json_str = super().serialize()
        json_obj = json.loads(json_str)
        json_obj["labels_file"] = self.labels_file
        return json.dumps(json_obj)

    def deserialize(self, target: str):
        try:
            json_obj = json.loads(target)
            if "labels_file" in json_obj:
                self.labels_file = json_obj["labels_file"]
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
