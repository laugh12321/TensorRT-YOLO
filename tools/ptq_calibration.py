#!/usr/bin/env python
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
# File    :   ptq_calibration.py
# Version :   2.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/03/24 17:48:00
# Desc    :   Quantize the model using specified configuration for calibration.
# ==============================================================================
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from typing import Any, Dict, Iterator

import numpy as np
import yaml
from easydict import EasyDict as edict

from tensorrt_yolo import ImageBatcher

with open(ROOT / "calibration.yaml", "r") as file:
    CONFIG = edict(yaml.safe_load(file))


def load_data() -> Iterator[Dict[str, Any]]:
    """
    Generator function to load data for calibration.

    Yields:
        Dict[str, Any]: A dictionary containing input data for calibration.
    """
    batcher = ImageBatcher(
        input_path=CONFIG.calibrator.data,
        batch_size=CONFIG.model.batch_shape[0],
        imgsz=CONFIG.model.batch_shape[2:],
        dynamic=False,
        dtype=np.float32,
        exact_batches=True,
    )
    for batch, images, batch_shape in batcher:
        yield {'images': batch}


if __name__ == '__main__':
    import subprocess as sp

    command = [
        "polygraphy",
        "convert",
        f"{CONFIG.model.input}",
        "--int8",
        "--precision-constraints",
        "prefer",
        "--sparse-weights",
        "--calib-base-cls",
        f"{CONFIG.calibrator.type}",
        "--data-loader-script",
        "ptq_calibration.py",
        "--calibration-cache",
        f"{CONFIG.calibrator.cache}",
        "-o",
        f"{CONFIG.model.output}",
    ]
    if CONFIG.model.dynamic:
        command.extend(
            [
                "--trt-min-shapes",
                f"{CONFIG.model.shapes.min}",
                "--trt-opt-shapes",
                f"{CONFIG.model.shapes.opt}",
                "--trt-max-shapes",
                f"{CONFIG.model.shapes.max}",
            ]
        )
    sp.run(command)
