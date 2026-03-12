"""OpenClawSolver: delegates to an OpenClaw agent via CLI subprocess.

OpenClaw is a multi-channel AI assistant with a full agentic runtime
(coding tools: read, write, edit, exec, browser, etc.).  This solver
wraps the ``openclaw agent`` CLI command and parses its JSON output.

OpenClaw manages its own model configuration (via ``~/.openclaw/openclaw.json``
or the ``NVIDIA_API_KEY`` env var).  The eval config's model service is used
only for the judge, not for OpenClaw's LLM calls.

Important: OpenClaw's file tools resolve paths relative to its *configured*
workspace, not the subprocess cwd.  The solver prepends the absolute workspace
path to the prompt so the agent uses correct file paths.  ``cwd`` is still set
to ``workspace_path`` so the ``exec`` tool runs commands in the right place.

Prerequisites:
  - Node.js >= 22
  - ``openclaw`` binary in PATH (or custom path via ``openclaw_bin``)
  - ``tools.fs.workspaceOnly`` must NOT be ``true`` in openclaw config
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0


def _check_prerequisites(openclaw_bin: str) -> None:
    """Validate that Node.js and openclaw are available."""
    import shutil
    import subprocess

    node = shutil.which("node")
    if not node:
        raise RuntimeError(
            "OpenClawSolver requires Node.js >= 22 but 'node' was not found in PATH. "
            "Install Node.js or activate your version manager (nvm, fnm, etc.)."
        )
    try:
        version_out = subprocess.check_output(
            [node, "--version"], text=True, timeout=10,
        ).strip()
        major = int(version_out.lstrip("v").split(".")[0])
        if major < 22:
            raise RuntimeError(
                f"OpenClawSolver requires Node.js >= 22 but found {version_out}."
            )
    except (subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Failed to check Node.js version: {exc}") from exc

    if not shutil.which(openclaw_bin):
        raise RuntimeError(
            f"OpenClawSolver: '{openclaw_bin}' not found in PATH. "
            "Install openclaw or provide the full path via sandbox.agent_cmd."
        )


def _extract_json(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from stdout, skipping any preamble.

    Node.js may emit experimental warnings or other text before the
    JSON payload that ``openclaw agent --json`` prints.
    """
    idx = raw.find("{")
    if idx < 0:
        return {}
    candidate = raw[idx:]
    # Find the matching closing brace (handles nested objects)
    depth = 0
    for i, ch in enumerate(candidate):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(candidate[: i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


class OpenClawSolver:
    """Solver that delegates to an OpenClaw agent via ``openclaw agent`` CLI.

    The solver spawns ``openclaw agent --message <prompt> --session-id <id>
    --json --local`` as an async subprocess, parses the JSON output, and
    returns the response text.

    For workspace-aware benchmarks (e.g. PinchBench), the absolute workspace
    path is prepended to the prompt so OpenClaw's file tools use the correct
    directory.  The subprocess ``cwd`` is also set to the workspace so that
    ``exec`` tool commands run in the right place.
    """

    def __init__(
        self,
        openclaw_bin: str = "openclaw",
        thinking: str = "high",
        timeout: float = _DEFAULT_TIMEOUT,
        *,
        skip_preflight: bool = False,
    ) -> None:
        self._bin = openclaw_bin
        self._thinking = thinking
        self._timeout = timeout
        if not skip_preflight:
            _check_prerequisites(openclaw_bin)

    async def solve(
        self, task: SeedResult, sandbox: Sandbox | None = None,
    ) -> SolveResult:
        workspace_path = task.metadata.get("workspace_path")
        task_id = task.metadata.get("task_id", "task")
        session_id = f"nel-{task_id}-{uuid.uuid4().hex[:8]}"

        if workspace_path:
            effective_prompt = (
                f"Your working directory for this task is: {workspace_path}\n"
                f"All file paths should be relative to or within this directory.\n\n"
                f"{task.prompt}"
            )
        else:
            effective_prompt = task.prompt

        cmd = [
            self._bin, "agent",
            "--message", effective_prompt,
            "--session-id", session_id,
            "--json",
            "--local",
        ]
        if self._thinking:
            cmd.extend(["--thinking", self._thinking])

        env = {**os.environ}
        cwd = workspace_path if workspace_path and Path(workspace_path).is_dir() else None

        logger.info("OpenClawSolver: session=%s workspace=%s", session_id, cwd or "(none)")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("OpenClawSolver: timeout after %.0fs for %s", self._timeout, task_id)
            return SolveResult(
                response="",
                trajectory=[{"error": "openclaw_timeout", "timeout": self._timeout}],
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            logger.warning(
                "OpenClawSolver: exit code %d for %s: %s",
                process.returncode, task_id, stderr[:500],
            )
            return SolveResult(
                response="",
                trajectory=[{
                    "error": "openclaw_exit",
                    "code": process.returncode,
                    "stderr": stderr[:2000],
                }],
            )

        output = _extract_json(stdout)
        if not output:
            logger.warning("OpenClawSolver: no JSON in stdout for %s", task_id)
            return SolveResult(
                response=stdout[:5000],
                trajectory=[{"warning": "no_json_output", "stderr": stderr[:2000]}],
            )

        payloads = output.get("payloads") or []
        response_text = "\n".join(
            p.get("text", "").strip()
            for p in payloads
            if p.get("text")
        )

        meta = output.get("meta", {})
        duration_ms = meta.get("durationMs", 0)

        trajectory: list[dict[str, Any]] = [{
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": response_text}],
            },
        }]

        logger.info(
            "OpenClawSolver: %s completed in %dms, %d chars response",
            task_id, duration_ms, len(response_text),
        )

        return SolveResult(response=response_text, trajectory=trajectory)

    async def close(self) -> None:
        pass
