#!/usr/bin/env python
# ==============================================================================
# Copyright (c) 2026 laugh12321 Authors. All Rights Reserved.
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
"""Shared dataclasses used across graph matching and rewriting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import onnx_graphsurgeon as gs


@dataclass(frozen=True)
class ModelInfo:
    """Static metadata inferred from the input ONNX model."""

    is_dynamic: bool
    batch_size: int


@dataclass
class GraphContext:
    """Graph state passed to handlers during matching and rewriting."""

    graph: gs.Graph
    model_info: ModelInfo
    score_thresh: float
    final_concat: gs.Node
    concat_output: gs.Variable
    mask_protos: Optional[gs.Variable] = None


@dataclass(frozen=True)
class MatchResult:
    """Scored description of why a handler matched a graph."""

    score: int
    reason: str


@dataclass(frozen=True)
class RewriteResult:
    """Summary of a successful graph rewrite."""

    handler_name: str
    plugin_op: Optional[str]
