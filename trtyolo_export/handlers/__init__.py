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
"""Handler registry for supported ONNX graph rewrite patterns."""

from __future__ import annotations

from typing import Optional, Sequence

from .handler_base import BaseHandler
from .handler_classify import ClassifyHandler
from .handler_legacy import (
    LegacyDetectUltralyticsHandler,
    LegacyDetectV3V5Handler,
    LegacyObbUltralyticsHandler,
    LegacyPoseUltralyticsHandler,
    LegacySegmentV3V5Handler,
    UltralyticsSegmentHandler,
)
from .handler_nms_free import (
    NmsFreeDetectHandler,
    NmsFreeObbHandler,
    NmsFreePoseHandler,
    NmsFreeSegmentHandler,
)

HANDLER_CLASSES = (
    ClassifyHandler,
    NmsFreeSegmentHandler,
    NmsFreeObbHandler,
    NmsFreePoseHandler,
    LegacyObbUltralyticsHandler,
    LegacyPoseUltralyticsHandler,
    UltralyticsSegmentHandler,
    LegacySegmentV3V5Handler,
    NmsFreeDetectHandler,
    LegacyDetectV3V5Handler,
    LegacyDetectUltralyticsHandler,
)


def default_handlers() -> Sequence[BaseHandler]:
    """Instantiate the default handler set in priority order."""
    return [handler_cls() for handler_cls in HANDLER_CLASSES]


def list_handler_names() -> Sequence[str]:
    """Return the registered handler names."""
    return [handler_cls.name for handler_cls in HANDLER_CLASSES]


def create_handlers(
    names: Optional[Sequence[str]] = None,
) -> Sequence[BaseHandler]:
    """Build handler instances from explicit names or the default registry."""
    if not names:
        return default_handlers()

    registry = {
        handler_cls.name: handler_cls for handler_cls in HANDLER_CLASSES
    }
    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(f"Unknown handlers: {', '.join(unknown)}")
    return [registry[name]() for name in names]


__all__ = [
    "BaseHandler",
    "ClassifyHandler",
    "LegacyDetectUltralyticsHandler",
    "LegacyDetectV3V5Handler",
    "LegacyObbUltralyticsHandler",
    "LegacyPoseUltralyticsHandler",
    "LegacySegmentV3V5Handler",
    "NmsFreeDetectHandler",
    "NmsFreeObbHandler",
    "NmsFreePoseHandler",
    "NmsFreeSegmentHandler",
    "UltralyticsSegmentHandler",
    "create_handlers",
    "default_handlers",
    "list_handler_names",
]
