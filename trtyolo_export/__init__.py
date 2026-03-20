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
"""Public package exports for the ONNX rewrite toolkit."""

__version__ = "2.0.0"

from importlib import import_module
from typing import TYPE_CHECKING, Any, List

if TYPE_CHECKING:
    from .handlers import (
        BaseHandler,
        create_handlers,
        default_handlers,
        list_handler_names,
    )
    from .pipeline import RewriteConfig, rewrite_onnx
    from .types import MatchResult, RewriteResult

_LAZY_IMPORTS = {
    "BaseHandler": (".handlers", "BaseHandler"),
    "MatchResult": (".types", "MatchResult"),
    "RewriteConfig": (".pipeline", "RewriteConfig"),
    "RewriteResult": (".types", "RewriteResult"),
    "create_handlers": (".handlers", "create_handlers"),
    "default_handlers": (".handlers", "default_handlers"),
    "list_handler_names": (".handlers", "list_handler_names"),
    "rewrite_onnx": (".pipeline", "rewrite_onnx"),
}

__all__ = [
    "__version__",
    "BaseHandler",
    "MatchResult",
    "RewriteConfig",
    "RewriteResult",
    "create_handlers",
    "default_handlers",
    "list_handler_names",
    "rewrite_onnx",
]


def __getattr__(name: str) -> Any:
    """Lazily import public package symbols on first access."""
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value


def __dir__() -> List[str]:
    """Return the package attributes exposed for interactive use."""
    return sorted(__all__)
