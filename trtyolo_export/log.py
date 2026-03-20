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
"""Logging helpers used by the CLI and rewrite pipeline."""

from __future__ import annotations

import sys

from loguru import logger as _logger

logger = _logger.bind(package="trtyolo_export")


def configure_logging(verbose: bool = True) -> None:
    """Configure a compact CLI-friendly logger."""
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        level="INFO" if verbose else "WARNING",
        format="<level>[{level.name[0]}]</level> <level>{message}</level>",
    )
