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
"""BYOB (Bring Your Own Agent) -- directory-based agent definition.

A ``ByobInstalledAgent`` reads install/run scripts from a directory so
users can define new agents without writing Python.  It plugs into
HarborSolver when ``harbor_agent`` points to a directory path.

Directory convention::

    my-agent/
    ├── agent.toml          # name, version, timeout, env_vars
    ├── install.sh.j2       # Jinja2 install template (same format Harbor uses)
    └── run.sh.j2           # Jinja2 run command template

All Harbor imports are lazy.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment as JinjaEnv
from jinja2 import select_autoescape

logger = logging.getLogger(__name__)

_jinja_env = JinjaEnv(autoescape=select_autoescape(default_for_string=False))


class ByobInstalledAgent:
    """Harbor ``BaseInstalledAgent``-compatible agent loaded from a directory.

    Instead of subclassing BaseInstalledAgent (which would force Harbor to
    be a hard dep at import time), this class duck-types the interface that
    HarborSolver calls: ``setup()``, ``run()``, ``populate_context_post_run()``.
    """

    def __init__(
        self,
        agent_dir: Path,
        logs_dir: Path,
        **kwargs: Any,
    ) -> None:
        self._agent_dir = agent_dir
        self._logs_dir = logs_dir
        self._kwargs = kwargs
        self._config = self._load_config()

    def _load_config(self) -> dict[str, Any]:
        toml_path = self._agent_dir / "agent.toml"
        if not toml_path.exists():
            return {"name": self._agent_dir.name, "timeout_sec": 1800}

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redefine]

        return tomllib.loads(toml_path.read_text())

    @property
    def _install_template(self) -> str | None:
        path = self._agent_dir / "install.sh.j2"
        return path.read_text() if path.exists() else None

    @property
    def _run_template(self) -> str:
        path = self._agent_dir / "run.sh.j2"
        if not path.exists():
            raise FileNotFoundError(f"BYOB agent requires run.sh.j2 in {self._agent_dir}")
        return path.read_text()

    async def setup(self, environment: Any) -> None:
        template_text = self._install_template
        if template_text is None:
            return

        template = _jinja_env.from_string(template_text)
        rendered = template.render(**self._kwargs)

        script_path = self._logs_dir / "install.sh"
        script_path.write_text(rendered)

        await environment.upload_file(
            source_path=script_path,
            target_path="/installed-agent/install.sh",
        )

        setup_env = {"DEBIAN_FRONTEND": "noninteractive"}
        setup_env.update(self._config.get("env", {}))

        result = await environment.exec(
            command="bash /installed-agent/install.sh",
            env=setup_env,
        )

        if result.return_code != 0:
            raise RuntimeError(
                f"BYOB agent setup failed (exit code {result.return_code}): {(result.stderr or '')[:500]}"
            )

    async def run(
        self,
        instruction: str,
        environment: Any,
        context: Any,
    ) -> None:
        template = _jinja_env.from_string(self._run_template)
        rendered = template.render(
            instruction=instruction,
            **self._kwargs,
        )

        run_env: dict[str, str] = {}
        run_env.update(self._config.get("env", {}))

        timeout = int(self._config.get("timeout_sec", 1800))

        command_dir = self._logs_dir / "command-0"
        command_dir.mkdir(parents=True, exist_ok=True)
        (command_dir / "command.txt").write_text(rendered)

        result = await environment.exec(
            command=rendered,
            env=run_env or None,
            timeout_sec=timeout,
        )

        self._last_stdout = result.stdout or ""

        (command_dir / "return-code.txt").write_text(str(result.return_code))
        if result.stdout:
            (command_dir / "stdout.txt").write_text(result.stdout)
        if result.stderr:
            (command_dir / "stderr.txt").write_text(result.stderr)

    def populate_context_post_run(self, context: Any) -> None:
        if self._last_stdout:
            if context.metadata is None:
                context.metadata = {}
            context.metadata["response"] = self._last_stdout
