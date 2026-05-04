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
"""Instruction-template resolution and rendering for benchmarks.

Templates are Jinja2 ``.md`` files.  Each benchmark config may specify an
``instruction_template`` that overrides the prompt produced by the
environment's ``seed()`` method.

Resolution order
~~~~~~~~~~~~~~~~
1. ``"off"`` / ``"none"`` / ``""``  →  no override.
2. Bare filename (e.g. ``swebench-instruction.md``)  →  look in the
   built-in ``templates/`` directory shipped with this package.
3. Absolute or relative path  →  use that file directly.
4. ``None``  →  no override.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import BaseLoader, Environment, TemplateNotFound, select_autoescape

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).resolve().parent


def resolve_template_path(
    explicit: str | None,
) -> Path | None:
    """Return the template file to use, or *None* for no override."""
    if explicit is None:
        return None

    low = explicit.strip().lower()
    if low in ("off", "none", ""):
        return None

    # Bare filename → look in the built-in templates directory first
    builtin = _TEMPLATES_DIR / explicit
    if builtin.is_file():
        return builtin

    p = Path(explicit)
    if not p.is_absolute():
        p = Path.cwd() / p
    if not p.exists():
        raise FileNotFoundError(f"instruction_template not found: {explicit!r} (checked built-in {builtin} and {p})")
    return p


class _PackageLoader(BaseLoader):
    """Jinja2 loader that resolves from an absolute *Path*."""

    def get_source(self, environment: Environment, template: str) -> tuple[str, str, Any]:
        p = Path(template)
        if not p.exists():
            raise TemplateNotFound(template)
        source = p.read_text(encoding="utf-8")
        return source, str(p), lambda: p.stat().st_mtime


_jinja_env = Environment(
    loader=_PackageLoader(),
    keep_trailing_newline=True,
    autoescape=select_autoescape(),
    auto_reload=False,
)


def render_template(
    template_path: Path,
    *,
    original_prompt: str,
    workspace_path: str,
    metadata: dict[str, Any],
) -> str:
    """Render *template_path* with the available seed context.

    Template variables
    ~~~~~~~~~~~~~~~~~~
    * ``workspace_path``  – sandbox workdir  (e.g. ``/testbed``, ``/app``)
    * ``original_prompt``  – raw prompt from the environment
    * ``instance``  – dict built from *metadata* plus ``problem_statement``
      (= *original_prompt*) and ``base_commit``
    * all top-level keys from *metadata* are also available directly
    """
    instance: dict[str, Any] = {
        **metadata,
        "problem_statement": original_prompt,
    }

    context: dict[str, Any] = {
        **metadata,
        "workspace_path": workspace_path,
        "original_prompt": original_prompt,
        "instance": instance,
    }

    tmpl = _jinja_env.get_template(str(template_path))
    return tmpl.render(context).strip()
