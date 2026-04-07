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
"""Docker-sandboxed code execution scorer.

Supports two modes:
  1. Sandbox protocol — if a ``Sandbox`` instance is passed, code is executed
     inside the running sandbox via ``sandbox.exec()``.
  2. Legacy fallback — if no sandbox is provided, ``docker run`` is spawned
     directly (original behavior, for backward compatibility).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from nemo_evaluator.scoring.types import ScorerInput

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox


def _extract_code(sample: ScorerInput) -> str:
    """Extract Python code from model response (markdown fences or raw)."""
    m = re.search(r"```(?:python)?\s*\n((?:\n|.)+?)```", sample.response, re.DOTALL)
    return m.group(1) if m else sample.response


def code_sandbox(sample: ScorerInput) -> dict[str, Any]:
    """Run code in a hardened Docker sandbox. Pipes code via stdin.

    Legacy synchronous path — used when no sandbox is available.
    """
    import subprocess

    prompt_code = sample.metadata.get("_prompt", "")
    test_code = sample.metadata.get("_test", "")
    entry_point = sample.metadata.get("_entry_point") or sample.metadata.get("entry_point", "solution")

    completion = _extract_code(sample)
    code = f"{prompt_code}{completion}\n{test_code}\ncheck({entry_point})"
    try:
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-i",
                "--network",
                "none",
                "--memory",
                "256m",
                "--cpus",
                "1",
                "--pids-limit",
                "64",
                "--read-only",
                "--tmpfs",
                "/tmp:size=64m",
                "--security-opt",
                "no-new-privileges",
                "python:3.12-slim",
                "python",
                "-",
            ],
            input=code,
            capture_output=True,
            text=True,
            timeout=30,
        )
        passed = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        passed = False

    return {"correct": passed, "extracted": completion[:500]}


async def code_sandbox_async(
    sample: ScorerInput,
    sandbox: Sandbox,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Run code inside an existing sandbox via the Sandbox protocol.

    The sandbox should already be started (e.g. with ``python:3.12-slim``).
    """
    prompt_code = sample.metadata.get("_prompt", "")
    test_code = sample.metadata.get("_test", "")
    entry_point = sample.metadata.get("_entry_point") or sample.metadata.get("entry_point", "solution")

    completion = _extract_code(sample)
    code = f"{prompt_code}{completion}\n{test_code}\ncheck({entry_point})"

    escaped = code.replace("'", "'\\''")
    result = await sandbox.exec(
        f"python -c '{escaped}'",
        timeout_sec=timeout,
    )
    passed = result.return_code == 0
    return {"correct": passed, "extracted": completion[:500]}
