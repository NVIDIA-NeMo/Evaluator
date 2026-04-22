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
"""OpenClawSolver: delegates to an OpenClaw agent via CLI or Docker sandbox.

OpenClaw is a multi-channel AI assistant with a full agentic runtime
(coding tools: read, write, edit, exec, browser, etc.).  This solver
wraps the ``openclaw agent`` CLI command and parses its JSON output.

Two execution modes:

1. **Local** (no sandbox): spawns ``openclaw agent`` as a subprocess on the
   host.  Requires Node.js >= 22 and ``openclaw`` in PATH.

2. **Sandbox** (Docker / Slurm / ECS): runs inside a per-problem container
   using the ``ghcr.io/openclaw/openclaw`` image.  The solver uploads a
   generated config and workspace files, then calls ``sandbox.exec()``.
   No host-side Node.js required.

Prerequisites (local mode only):
  - Node.js >= 22
  - ``openclaw`` binary in PATH (or custom path via ``openclaw_bin``)
  - ``tools.fs.workspaceOnly`` must NOT be ``true`` in openclaw config
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult
from .trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0

_OPENCLAW_BIN = "/app/openclaw.mjs"
_OPENCLAW_CONFIG_DIR = "/home/node/.openclaw"
_CONTAINER_WORKSPACE = "/home/node/workspace"
_CONTAINER_SESSIONS_DIR = f"{_OPENCLAW_CONFIG_DIR}/agents/main/sessions"

_DEFAULT_CONTEXT_WINDOW = 131_072
_DEFAULT_MAX_TOKENS = 16_384
_DEFAULT_MAX_CONCURRENT = 4


def _build_openclaw_config(
    model_url: str,
    model_id: str,
    *,
    api_key: str = "",
    context_window: int = _DEFAULT_CONTEXT_WINDOW,
    max_tokens: int | None = None,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> dict[str, Any]:
    """Build an OpenClaw config for an OpenAI-compatible provider.

    Provider name is derived from the ``model_id`` prefix (e.g. ``nvidia``
    from ``nvidia/nemotron-...``).  Sampling params are injected per-request
    by the adapter proxy's ``payload_modifier``, not baked into the config.

    When ``api_key`` is empty we deliberately omit ``apiKey`` from the
    provider entry so OpenClaw's auth resolver falls through to its
    synthetic local-key path for localhost ``baseUrl``s (the adapter proxy
    reached via SSH reverse tunnel).  Writing a placeholder like
    ``"NVIDIA_API_KEY"`` trips ``hasExplicitProviderApiKeyConfig`` in
    ``model-auth.ts`` and disables the synthetic path, causing
    ``No API key found for provider "custom"`` at agent start.
    """
    provider_name = model_id.split("/")[0] if "/" in model_id else "custom"
    openclaw_model_ref = f"{provider_name}/{model_id}"

    model_entry: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "input": ["text"],
        "contextWindow": context_window,
    }
    if max_tokens is not None:
        model_entry["maxTokens"] = max_tokens

    agent_defaults: dict[str, Any] = {
        "model": {"primary": openclaw_model_ref},
        "workspace": _CONTAINER_WORKSPACE,
        "skipBootstrap": True,
        "compaction": {"mode": "safeguard"},
        "maxConcurrent": max_concurrent,
    }

    provider_entry: dict[str, Any] = {
        "baseUrl": model_url,
        "api": "openai-completions",
        "models": [model_entry],
    }
    if api_key:
        provider_entry["apiKey"] = api_key

    return {
        "models": {
            "mode": "merge",
            "providers": {provider_name: provider_entry},
        },
        "agents": {"defaults": agent_defaults},
        "tools": {"fs": {"workspaceOnly": False}},
    }


_TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".py",
        ".js",
        ".ts",
        ".json",
        ".yaml",
        ".yml",
        ".csv",
        ".html",
        ".css",
        ".ics",
        ".xml",
        ".toml",
        ".cfg",
        ".ini",
        ".sh",
        ".bash",
        ".log",
        ".rst",
        ".tex",
    }
)
_MAX_FILE_SIZE = 50_000


def _read_workspace_files(
    workspace: Path,
    pre_existing: set[str] | None = None,
) -> str:
    """Read *new* text files from the workspace and format them for the response.

    Agent solvers create files that automated graders check directly.
    But the LLM-as-judge pipeline only sees ``SolveResult.response``.
    Including file contents in the response lets the judge evaluate
    the actual deliverable, not just the agent's summary.

    ``pre_existing`` is the set of relative paths that existed before the
    agent ran.  Files in that set and anything under ``.openclaw/`` are
    skipped so that task inputs don't pollute the response.  OpenClaw's
    workspace boilerplate is suppressed at the source via the
    ``skipBootstrap`` config flag.
    """
    if pre_existing is None:
        pre_existing = set()

    parts: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        rel_str = str(rel)
        if rel_str.startswith(".openclaw"):
            continue
        if rel_str in pre_existing:
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        if path.stat().st_size > _MAX_FILE_SIZE:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").rstrip()
        except OSError:
            continue
        parts.append(f"--- {rel} ---\n{content}")
    return "\n\n".join(parts)


def _check_prerequisites(openclaw_bin: str) -> None:
    """Validate that Node.js and openclaw are available on the host."""
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
            [node, "--version"],
            text=True,
            timeout=10,
        ).strip()
        major = int(version_out.lstrip("v").split(".")[0])
        if major < 22:
            raise RuntimeError(f"OpenClawSolver requires Node.js >= 22 but found {version_out}.")
    except (subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Failed to check Node.js version: {exc}") from exc

    if not shutil.which(openclaw_bin):
        raise RuntimeError(
            f"OpenClawSolver: '{openclaw_bin}' not found in PATH. "
            "Install openclaw or provide the full path via sandbox.agent_cmd."
        )


_MAX_SCAN_BYTES = 4_000_000
_JSON_DECODER = json.JSONDecoder()


def _iter_json_objects(raw: str) -> Iterator[dict[str, Any]]:
    """Yield every top-level JSON object embedded in ``raw``, in order.

    Uses :meth:`json.JSONDecoder.raw_decode` so the inner loop runs in C
    and string literals containing ``{`` / ``}`` are handled correctly.
    On a failed parse we advance one character and retry, which makes the
    scanner tolerant of preamble (Node.js experimental warnings, OpenClaw
    ``[subsystem] ...`` log lines, etc.).  Non-object JSON values are
    skipped so we never latch onto a bare array or scalar.
    """
    pos = 0
    n = len(raw)
    while pos < n:
        start = raw.find("{", pos)
        if start < 0:
            return
        try:
            obj, end = _JSON_DECODER.raw_decode(raw, start)
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(obj, dict):
            yield obj
        pos = max(end, start + 1)


def _extract_openclaw_envelope(stdout: str, stderr: str) -> dict[str, Any]:
    """Locate the ``openclaw agent --json`` result envelope.

    Prefers ``stdout`` (the documented channel) and falls back to ``stderr``.
    The stderr fallback is **not** transitional — it is load-bearing:
    upstream OpenClaw misroutes the envelope to stderr whenever ``--json``
    mode flips ``loggingState.forceConsoleToStderr``, because
    ``src/agents/command/delivery.ts`` emits via ``runtime.log`` (patched
    ``console.log``) instead of ``writeRuntimeJson`` (direct
    ``process.stdout.write``).  The dedicated ``writeRuntimeJson`` API
    exists in ``src/runtime.ts`` but was never wired into the ``--local``
    envelope emission.  We intentionally do not patch upstream, so this
    logic stays.

    We pick the *last* object containing a ``payloads`` key on each stream
    rather than the first, so that any future pre-envelope diagnostic log
    which happens to structurally include ``payloads`` cannot hijack the
    result.  By construction the real envelope is the last thing
    ``delivery.ts`` emits before command return.

    Each stream is tail-capped to :data:`_MAX_SCAN_BYTES` to bound parse
    time on multi-MB traces (``raw_decode`` is fast, but callers should
    not be able to feed us arbitrary input).
    """
    for raw in (stdout, stderr):
        if not raw:
            continue
        if len(raw) > _MAX_SCAN_BYTES:
            raw = raw[-_MAX_SCAN_BYTES:]
        last: dict[str, Any] | None = None
        for obj in _iter_json_objects(raw):
            if "payloads" in obj:
                last = obj
        if last is not None:
            return last
    return {}


def _parse_response(output: dict[str, Any]) -> tuple[str, ModelResponse, list[dict[str, Any]]]:
    """Extract response text, model response, and trajectory from openclaw JSON output."""
    payloads = output.get("payloads") or []
    response_text = "\n".join(p.get("text", "").strip() for p in payloads if p.get("text"))
    meta = output.get("meta", {})
    duration_ms = meta.get("durationMs", 0)
    agent_meta = meta.get("agentMeta", {})

    usage = meta.get("usage", {})
    prompt_tokens = usage.get("inputTokens", 0) or usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("outputTokens", 0) or usage.get("completion_tokens", 0)
    total_tokens = prompt_tokens + completion_tokens
    if total_tokens == 0 and response_text:
        total_tokens = len(response_text) // 4

    model_response = ModelResponse(
        content=response_text,
        model=agent_meta.get("model", ""),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=float(duration_ms),
    )

    steps: list[dict[str, Any]] = []
    for p in payloads:
        msg_text = p.get("text", "")
        step: dict[str, Any] = {
            "source": "agent",
            "message": msg_text,
        }
        extras: dict[str, Any] = {}
        if p.get("mediaUrl"):
            extras["media_urls"] = [p["mediaUrl"]]
        if p.get("mediaUrls"):
            extras.setdefault("media_urls", []).extend(p["mediaUrls"])
        if p.get("isError"):
            extras["is_error"] = True
        if extras:
            step["extra"] = extras
        if msg_text or extras:
            steps.append(step)

    oc_extra: dict[str, Any] | None = None
    if agent_meta:
        oc_extra = {
            "provider": agent_meta.get("provider", ""),
            "model": agent_meta.get("model", ""),
            "duration_ms": duration_ms,
            "openclaw_session_id": agent_meta.get("sessionId", ""),
        }

    return response_text, model_response, steps, oc_extra


def _parse_session_jsonl(raw: str) -> list[dict[str, Any]]:
    """Parse an OpenClaw session JSONL transcript into ATIF step dicts."""
    steps: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        role = record.get("role", "")
        if role == "assistant":
            content = record.get("content", [])
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            for block in content if isinstance(content, list) else []:
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif btype == "tool_use":
                    tool_calls.append(
                        {
                            "tool_call_id": block.get("id", f"call_{len(steps)}"),
                            "function_name": block.get("name", ""),
                            "arguments": block.get("input", {}),
                        }
                    )
            step: dict[str, Any] = {
                "source": "agent",
                "message": "\n".join(text_parts),
            }
            if tool_calls:
                step["tool_calls"] = tool_calls
            if text_parts or tool_calls:
                steps.append(step)
        elif role == "tool":
            content = record.get("content", "")
            if isinstance(content, list):
                content = "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
            if steps and steps[-1].get("source") == "agent":
                steps[-1].setdefault("observation", {"results": []})
                steps[-1]["observation"]["results"].append(
                    {
                        "content": str(content)[:2000],
                        "source_call_id": record.get("tool_use_id"),
                    }
                )
            else:
                steps.append(
                    {
                        "source": "system",
                        "message": str(content)[:2000],
                        "extra": {"tool_use_id": record.get("tool_use_id", "")},
                    }
                )
    return steps


class OpenClawSolver:
    """Solver that delegates to an OpenClaw agent.

    Runs inside a container (Docker / Slurm / ECS) when a sandbox is
    configured, otherwise falls back to a local subprocess.

    An OpenClaw config is generated from ``model_url``, ``model_id``,
    and the tuning knobs (``context_window``, ``max_tokens``,
    ``max_concurrent``).  Set ``config_path`` to sideload a fully custom
    ``openclaw.json`` instead.
    """

    def __init__(
        self,
        openclaw_bin: str = "openclaw",
        thinking: str = "high",
        timeout: float = _DEFAULT_TIMEOUT,
        model_url: str = "",
        model_id: str = "",
        api_key: str | None = None,
        context_window: int = _DEFAULT_CONTEXT_WINDOW,
        max_tokens: int | None = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        config_path: str | None = None,
        *,
        skip_preflight: bool = False,
    ) -> None:
        self._bin = openclaw_bin
        self._thinking = thinking
        self._timeout = timeout
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._context_window = context_window
        self._max_tokens = max_tokens
        self._max_concurrent = max_concurrent
        self._config_path = Path(config_path) if config_path else None
        if self._config_path and not self._config_path.is_file():
            raise FileNotFoundError(f"openclaw_config points to {self._config_path} which does not exist")
        if not skip_preflight:
            _check_prerequisites(openclaw_bin)

    async def solve(
        self,
        task: SeedResult,
        sandbox: Sandbox | None = None,
    ) -> SolveResult:
        if sandbox is not None:
            return await self._solve_sandbox(task, sandbox)
        return await self._solve_local(task)

    # ------------------------------------------------------------------
    # Sandbox path: run inside a Docker / Slurm / ECS container
    # ------------------------------------------------------------------

    async def _solve_sandbox(self, task: SeedResult, sandbox: Sandbox) -> SolveResult:
        task_id = task.metadata.get("task_id", "task")
        session_id = f"nel-{task_id}-{uuid.uuid4().hex[:8]}"
        workspace_path = task.metadata.get("workspace_path")

        logger.info("OpenClawSolver[sandbox]: session=%s task=%s", session_id, task_id)

        resolved_model_url = sandbox.resolved_endpoint_url("MODEL_BASE_URL") or (
            sandbox.resolve_outside_endpoint(self._model_url) if self._model_url else self._model_url
        )
        await self._setup_config(sandbox, model_url_override=resolved_model_url)

        pre_existing_files: set[str] = set()
        if workspace_path and Path(workspace_path).is_dir():
            await self._upload_workspace(sandbox, Path(workspace_path))
            pre_existing_files = {
                str(p.relative_to(workspace_path)) for p in Path(workspace_path).rglob("*") if p.is_file()
            }

        effective_prompt = self._build_prompt(task.prompt, _CONTAINER_WORKSPACE if workspace_path else None)

        prompt_file = "/tmp/nel_prompt.txt"
        await self._write_file(sandbox, prompt_file, effective_prompt)

        cmd_parts = [
            f"node {_OPENCLAW_BIN} agent",
            "--agent main",
            f'--message "$(cat {prompt_file})"',
            f"--session-id {shlex.quote(session_id)}",
            "--json",
            "--local",
        ]
        if self._thinking:
            cmd_parts.append(f"--thinking {shlex.quote(self._thinking)}")

        env_prefix = ""
        if self._api_key:
            env_prefix = f"NVIDIA_API_KEY={shlex.quote(self._api_key)} "

        inner_cmd = f"{env_prefix}{' '.join(cmd_parts)}"
        full_cmd = f"bash -c {shlex.quote(inner_cmd)}"

        result = await sandbox.exec(full_cmd, timeout_sec=self._timeout)

        if workspace_path and Path(workspace_path).is_dir():
            await self._download_workspace(sandbox, Path(workspace_path))

        if result.return_code != 0:
            err_msg = f"exit code {result.return_code}: {result.stderr[:500]}"
            logger.warning("OpenClawSolver[sandbox]: %s for %s", err_msg, task_id)
            return SolveResult(
                response="",
                trajectory=build_atif_trajectory(
                    [{"source": "system", "message": result.stderr[:2000]}],
                    agent_name="openclaw",
                    status="error",
                    extra={"exit_code": result.return_code},
                ),
                error=err_msg,
            )

        output = _extract_openclaw_envelope(result.stdout, result.stderr)
        if not output:
            err_msg = f"no JSON output for {task_id}"
            logger.warning("OpenClawSolver[sandbox]: %s", err_msg)
            return SolveResult(
                response=(result.stdout or result.stderr)[:5000],
                trajectory=build_atif_trajectory(
                    [{"source": "system", "message": result.stderr.strip()[:2000] or "no JSON output"}],
                    agent_name="openclaw",
                    status="error",
                ),
                error=err_msg,
            )

        response_text, model_response, steps, oc_extra = _parse_response(output)

        if workspace_path:
            file_addendum = _read_workspace_files(Path(workspace_path), pre_existing_files)
            if file_addendum:
                response_text = f"{response_text}\n\n{file_addendum}" if response_text else file_addendum

        transcript_steps = await self._read_session_transcript(sandbox, session_id)
        if transcript_steps:
            steps = transcript_steps + steps

        trajectory = build_atif_trajectory(
            steps,
            agent_name="openclaw",
            model_name=model_response.model or None,
            prompt_tokens=model_response.prompt_tokens or 0,
            completion_tokens=model_response.completion_tokens or 0,
            extra=oc_extra,
        )

        logger.info(
            "OpenClawSolver[sandbox]: %s completed in %dms, %d tok, %d steps",
            task_id,
            model_response.latency_ms,
            model_response.total_tokens,
            len(steps),
        )
        return SolveResult(
            response=response_text,
            model_response=model_response,
            trajectory=trajectory,
        )

    async def _read_session_transcript(
        self,
        sandbox: Sandbox,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Read the OpenClaw session JSONL from the container."""
        import base64

        remote_path = f"{_CONTAINER_SESSIONS_DIR}/{session_id}.jsonl"
        try:
            result = await sandbox.exec(f"base64 {shlex.quote(remote_path)} 2>/dev/null")
            if result.return_code != 0 or not result.stdout.strip():
                return []
            raw = base64.b64decode(result.stdout.strip()).decode("utf-8", errors="replace")
        except Exception:
            return []
        return _parse_session_jsonl(raw)

    async def _setup_config(self, sandbox: Sandbox, *, model_url_override: str | None = None) -> None:
        """Write an OpenClaw config into the container.

        Uses ``config_path`` verbatim when provided, otherwise generates
        one from the solver parameters.  ``model_url_override`` takes
        precedence over ``self._model_url`` (used for session-scoped URLs).
        """
        if self._config_path is not None:
            config_json = self._config_path.read_text(encoding="utf-8")
        else:
            config = _build_openclaw_config(
                model_url=model_url_override or self._model_url,
                model_id=self._model_id,
                api_key=self._api_key or "",
                context_window=self._context_window,
                max_tokens=self._max_tokens,
                max_concurrent=self._max_concurrent,
            )
            config_json = json.dumps(config, indent=2)
        await self._write_file(sandbox, f"{_OPENCLAW_CONFIG_DIR}/openclaw.json", config_json)

    async def _upload_workspace(self, sandbox: Sandbox, host_workspace: Path) -> None:
        """Copy workspace files from host into the container.

        Handles both text and binary files (e.g. PDFs) correctly.
        """
        await sandbox.exec(f"mkdir -p {_CONTAINER_WORKSPACE}")
        for item in host_workspace.rglob("*"):
            if item.is_file():
                rel = item.relative_to(host_workspace)
                remote = f"{_CONTAINER_WORKSPACE}/{rel}"
                data = item.read_bytes()
                await self._write_file(sandbox, remote, data)

    async def _download_workspace(self, sandbox: Sandbox, host_workspace: Path) -> None:
        """Sync container workspace files back to the host.

        The PinchBench grader (and others) inspect files in the host
        workspace after the agent runs.  Without this step, any files
        the agent created inside the container would be invisible to
        the scorer.

        The generated openclaw.json sets ``agents.defaults.workspace``
        to ``_CONTAINER_WORKSPACE`` so that OpenClaw's internal
        ``process.chdir()`` and file tools target the same directory
        we search here.
        """
        import base64

        listing = await sandbox.exec(f"find {_CONTAINER_WORKSPACE} -type f 2>/dev/null || true")
        remote_files = [
            line.strip()
            for line in listing.stdout.splitlines()
            if line.strip() and line.strip().startswith(_CONTAINER_WORKSPACE)
        ]
        if not remote_files:
            logger.warning(
                "No files found in container workspace %s; agent may have written elsewhere",
                _CONTAINER_WORKSPACE,
            )
        for remote_path in remote_files:
            rel = remote_path[len(_CONTAINER_WORKSPACE) + 1 :]
            if not rel:
                continue
            local_path = host_workspace / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = await sandbox.exec(f"base64 {shlex.quote(remote_path)}")
                if result.return_code == 0 and result.stdout.strip():
                    data = base64.b64decode(result.stdout.strip())
                    local_path.write_bytes(data)
            except Exception as exc:
                logger.debug("Failed to download %s: %s", remote_path, exc)

        logger.debug("Downloaded %d files to %s", len(remote_files), host_workspace)

    _WRITE_CHUNK_SIZE = 65_536

    @staticmethod
    async def _write_file(
        sandbox: Sandbox,
        remote_path: str,
        content: str | bytes,
    ) -> None:
        """Write a file inside the container via base64-encoded exec.

        Files are created by the container user (avoids ``docker cp``
        ownership issues with the non-root ``node`` user).

        Large payloads are split into chunks to stay under the OS
        argument-length limit for ``docker exec``.
        """
        import base64

        raw = content if isinstance(content, bytes) else content.encode()
        remote_dir = str(Path(remote_path).parent)
        await sandbox.exec(f"mkdir -p {shlex.quote(remote_dir)}")

        chunk_size = OpenClawSolver._WRITE_CHUNK_SIZE
        if len(raw) <= chunk_size:
            b64 = base64.b64encode(raw).decode()
            await sandbox.exec(f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}")
            return

        for i in range(0, len(raw), chunk_size):
            chunk_b64 = base64.b64encode(raw[i : i + chunk_size]).decode()
            op = ">" if i == 0 else ">>"
            await sandbox.exec(f"echo {shlex.quote(chunk_b64)} | base64 -d {op} {shlex.quote(remote_path)}")

    # ------------------------------------------------------------------
    # Local path: run as a host subprocess (original behavior)
    # ------------------------------------------------------------------

    def _prepare_local_config(self, env: dict[str, str]) -> Path | None:
        """Stage a custom OpenClaw config for local execution.

        Creates a temporary HOME with ``~/.openclaw/openclaw.json`` so the
        user's own config is not overwritten.  Returns the temp dir (caller
        must keep a reference to prevent GC) or *None* when no override.
        """
        if self._config_path is None:
            return None
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="nel-openclaw-"))
        dest = tmp / ".openclaw"
        dest.mkdir()
        import shutil

        shutil.copy2(self._config_path, dest / "openclaw.json")
        env["HOME"] = str(tmp)
        return tmp

    async def _solve_local(self, task: SeedResult) -> SolveResult:
        task_id = task.metadata.get("task_id", "task")
        session_id = f"nel-{task_id}-{uuid.uuid4().hex[:8]}"
        workspace_path = task.metadata.get("workspace_path")

        pre_existing_files: set[str] = set()
        if workspace_path and Path(workspace_path).is_dir():
            pre_existing_files = {
                str(p.relative_to(workspace_path)) for p in Path(workspace_path).rglob("*") if p.is_file()
            }

        effective_prompt = self._build_prompt(task.prompt, workspace_path)

        cmd = [
            self._bin,
            "agent",
            "--message",
            effective_prompt,
            "--session-id",
            session_id,
            "--json",
            "--local",
        ]
        if self._thinking:
            cmd.extend(["--thinking", self._thinking])

        env = {**os.environ}
        if self._api_key:
            env["NVIDIA_API_KEY"] = self._api_key
        _tmp_home = self._prepare_local_config(env)
        cwd = workspace_path if workspace_path and Path(workspace_path).is_dir() else None

        logger.info("OpenClawSolver[local]: session=%s workspace=%s", session_id, cwd or "(none)")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            err_msg = f"Timeout after {self._timeout}s"
            logger.warning("OpenClawSolver[local]: %s for %s", err_msg, task_id)
            return SolveResult(
                response="",
                trajectory=build_atif_trajectory(
                    [{"source": "system", "message": err_msg}],
                    agent_name="openclaw",
                    status="error",
                ),
                error=err_msg,
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            err_msg = f"exit code {process.returncode}: {stderr[:500]}"
            logger.warning("OpenClawSolver[local]: %s for %s", err_msg, task_id)
            return SolveResult(
                response="",
                trajectory=build_atif_trajectory(
                    [{"source": "system", "message": stderr[:2000]}],
                    agent_name="openclaw",
                    status="error",
                    extra={"exit_code": process.returncode},
                ),
                error=err_msg,
            )

        output = _extract_openclaw_envelope(stdout, stderr)
        if not output:
            err_msg = f"no JSON output for {task_id}"
            logger.warning("OpenClawSolver[local]: %s", err_msg)
            return SolveResult(
                response=(stdout or stderr)[:5000],
                trajectory=build_atif_trajectory(
                    [{"source": "system", "message": stderr.strip()[:2000] or "no JSON output"}],
                    agent_name="openclaw",
                    status="error",
                ),
                error=err_msg,
            )

        response_text, model_response, steps, oc_extra = _parse_response(output)

        if workspace_path:
            file_addendum = _read_workspace_files(Path(workspace_path), pre_existing_files)
            if file_addendum:
                response_text = f"{response_text}\n\n{file_addendum}" if response_text else file_addendum

        home = env.get("HOME", str(Path.home()))
        transcript_steps = self._read_local_session_transcript(home, session_id)
        if transcript_steps:
            steps = transcript_steps + steps

        trajectory = build_atif_trajectory(
            steps,
            agent_name="openclaw",
            model_name=model_response.model or None,
            prompt_tokens=model_response.prompt_tokens or 0,
            completion_tokens=model_response.completion_tokens or 0,
            extra=oc_extra,
        )

        logger.info(
            "OpenClawSolver[local]: %s completed in %dms, %d tok, %d steps",
            task_id,
            model_response.latency_ms,
            model_response.total_tokens,
            len(steps),
        )
        return SolveResult(
            response=response_text,
            model_response=model_response,
            trajectory=trajectory,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_local_session_transcript(
        home: str,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Read a session transcript from the local filesystem."""
        sessions_dir = Path(home) / ".openclaw" / "agents" / "main" / "sessions"
        transcript_path = sessions_dir / f"{session_id}.jsonl"
        if not transcript_path.exists():
            return []
        return _parse_session_jsonl(transcript_path.read_text(encoding="utf-8", errors="replace"))

    @staticmethod
    def _build_prompt(raw_prompt: str, workspace_path: str | None) -> str:
        if workspace_path:
            return (
                f"Your working directory for this task is: {workspace_path}\n"
                f"All file paths should be relative to or within this directory.\n\n"
                f"{raw_prompt}"
            )
        return raw_prompt

    async def close(self) -> None:
        pass
