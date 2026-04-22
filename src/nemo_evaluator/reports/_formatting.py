# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared terminal formatting utilities for text renderers.

Provides ANSI-color helpers via ``click.style`` with ``NO_COLOR`` support,
verdict-color mapping, and a simple progress bar builder.
"""

from __future__ import annotations

import os

import click

NO_COLOR = os.environ.get("NO_COLOR") is not None


def style(text: str, **kwargs) -> str:
    """Wrap *text* in ANSI color/bold via ``click.style``, respecting ``NO_COLOR``."""
    if NO_COLOR:
        return text
    return click.style(text, **kwargs)


def bar(value: float, width: int = 20) -> str:
    """Return a block-char progress bar representing *value* (0-1)."""
    filled = round(value * width)
    return "\u2588" * filled + "\u2591" * (width - filled)


def verdict_color(verdict: str) -> str:
    """Map a verdict string to a Click color name."""
    if verdict in ("BLOCK", "NO-GO"):
        return "red"
    if verdict in ("WARN", "INCONCLUSIVE"):
        return "yellow"
    return "green"
