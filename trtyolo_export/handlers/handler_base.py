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
"""Base interfaces shared by all graph rewrite handlers."""

from __future__ import annotations

from typing import Optional

from ..types import GraphContext, MatchResult, RewriteResult


class BaseHandler:
    """Base interface for graph pattern matching and conversion."""

    name = "base"

    def match(self, ctx: GraphContext) -> Optional[MatchResult]:
        """Return a match result when the handler recognizes the graph."""
        raise NotImplementedError

    def rewrite(self, ctx: GraphContext) -> RewriteResult:
        """Rewrite the matched graph and describe the produced outputs."""
        raise NotImplementedError
