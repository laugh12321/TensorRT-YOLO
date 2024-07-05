#!/usr/bin/env python
# -*-coding:utf-8 -*-
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
# File    :   timer.py
# Version :   1.0
# Author  :   laugh12321
# Contact :   laugh12321@vip.qq.com
# Date    :   2024/07/05 13:39:01
# Desc    :   Python bindings for CpuTimer and GpuTimer using Pybind11.
# ==============================================================================
from .. import c_lib_wrap as C

__all__ = ["CpuTimer", "GpuTimer"]


class CpuTimer:
    """Class to interface with the CpuTimer implemented in C++."""

    def __init__(self) -> None:
        """
        Initialize the CpuTimer.
        """
        self._timer = C.timer.CpuTimer()

    def start(self) -> None:
        """
        Start the CPU timer.
        """
        self._timer.start()

    def stop(self) -> None:
        """
        Stop the CPU timer.
        """
        self._timer.stop()

    def reset(self) -> None:
        """
        Reset the CPU timer.
        """
        self._timer.reset()

    def seconds(self) -> float:
        """Get the elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds.
        """
        return self._timer.seconds()

    def milliseconds(self) -> float:
        """Get the elapsed time in milliseconds.

        Returns:
            float: Elapsed time in milliseconds.
        """
        return self._timer.milliseconds()

    def microseconds(self) -> float:
        """Get the elapsed time in microseconds.

        Returns:
            float: Elapsed time in microseconds.
        """
        return self._timer.microseconds()


class GpuTimer:
    """Class to interface with the GpuTimer implemented in C++."""

    def __init__(self) -> None:
        """
        Initialize the GpuTimer.
        """
        self._timer = C.timer.GpuTimer()

    def start(self) -> None:
        """
        Start the GPU timer.
        """
        self._timer.start()

    def stop(self) -> None:
        """
        Stop the GPU timer.
        """
        self._timer.stop()

    def reset(self) -> None:
        """
        Reset the GPU timer.
        """
        self._timer.reset()

    def seconds(self) -> float:
        """
        Get the elapsed time in seconds.

        Returns:
            float: Elapsed time in seconds.
        """
        return self._timer.seconds()

    def milliseconds(self) -> float:
        """
        Get the elapsed time in milliseconds.

        Returns:
            float: Elapsed time in milliseconds.
        """
        return self._timer.milliseconds()

    def microseconds(self) -> float:
        """
        Get the elapsed time in microseconds.

        Returns:
            float: Elapsed time in microseconds.
        """
        return self._timer.microseconds()
