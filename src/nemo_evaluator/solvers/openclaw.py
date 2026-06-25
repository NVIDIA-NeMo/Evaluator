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
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import ErrorKind, SolveResult
from .openclaw_helpers import (
    _CONTAINER_SESSIONS_DIR,
    _CONTAINER_WORKSPACE,
    _OPENCLAW_CONFIG_DIR,
    _coerce_timeout,
    _container_session_find_command,
    _container_workspace_find_command,
    _decode_b64_text,
    _extract_openclaw_envelope,
    _finalize_openclaw_solve,
    _format_timeout_seconds,
    _iter_workspace_files,
    _openclaw_error_kind,
    _parse_response,
    _redact_secret,
    _resolve_container_session_path,
    _resolve_local_session_path,
    _sandbox_agent_exec_timeout,
    _session_path_candidates_from_index,
    _unique_strings,
    _workspace_rel_is_skipped,
)
from .trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0

_OPENCLAW_BIN = "/app/openclaw.mjs"
_DEFAULT_CONTEXT_WINDOW: int | None = None
_DEFAULT_IDLE_TIMEOUT_SECONDS = 600
_DEFAULT_MAX_TOKENS = 16_384
_DEFAULT_MAX_CONCURRENT = 4
_OPENCLAW_AGENT_TIMEOUT_KILL_AFTER_SECONDS = 5.0
_OPENCLAW_AGENT_TIMEOUT_SENTINEL = "__NEL_OPENCLAW_AGENT_TIMEOUT__"


def _normalize_task_sessions(task: SeedResult) -> list[dict[str, Any]]:
    """Normalize PinchBench task sessions into OpenClaw turns.

    PinchBench frontmatter can provide several prompts for one task and
    mark a turn with ``new_session`` when the upstream runner should reset
    OpenClaw conversation state while preserving workspace files.
    """
    sessions = task.metadata.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        return [{"id": "default", "prompt": task.prompt, "new_session": False}]

    normalized: list[dict[str, Any]] = []
    for idx, session in enumerate(sessions):
        if not isinstance(session, dict):
            continue
        prompt = session.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        session_id = session.get("id")
        normalized.append(
            {
                "id": str(session_id) if session_id else f"session_{idx + 1}",
                "prompt": prompt,
                "new_session": bool(session.get("new_session", False)),
            }
        )
    return normalized or [{"id": "default", "prompt": task.prompt, "new_session": False}]


def _metadata_requires_fws(metadata: dict[str, Any]) -> bool:
    prerequisites = metadata.get("prerequisites") or []
    return isinstance(prerequisites, list) and any(
        isinstance(item, str) and "fws" in item.lower() for item in prerequisites
    )


def _is_pinchbench_task(metadata: dict[str, Any]) -> bool:
    return metadata.get("source") == "pinchbench" or "pinchbench_source_ref" in metadata


def _build_fws_env_setup(metadata: dict[str, Any]) -> str:
    if not _metadata_requires_fws(metadata):
        return ""
    prerequisites = metadata.get("prerequisites") or []
    has_gh_cli = isinstance(prerequisites, list) and any(
        isinstance(item, str) and item.strip().lower() == "cli:gh" for item in prerequisites
    )
    setup = r"""
if ! command -v fws >/dev/null 2>&1; then
  echo "PinchBench task requires fws, but fws is not installed in this sandbox." >&2
  exit 127
fi
if ! fws server status 2>/dev/null | grep -q "Server running"; then
  if ! fws server start >/tmp/nel-fws-start.log 2>&1; then
    cat /tmp/nel-fws-start.log >&2 || true
    exit 127
  fi
fi
_nel_fws_env="$(fws server env 2>/dev/null || true)"
if [ -n "$_nel_fws_env" ]; then
  eval "$_nel_fws_env"
else
  export GOOGLE_WORKSPACE_CLI_CONFIG_DIR="${GOOGLE_WORKSPACE_CLI_CONFIG_DIR:-$HOME/.local/share/fws/config}"
  export GOOGLE_WORKSPACE_CLI_TOKEN="${GOOGLE_WORKSPACE_CLI_TOKEN:-fake}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://localhost:4101}"
  _nel_fws_cert="$HOME/.local/share/fws/certs/ca-bundle.crt"
  if [ ! -f "$_nel_fws_cert" ]; then
    _nel_fws_cert="$HOME/.local/share/fws/certs/ca.crt"
  fi
  export SSL_CERT_FILE="${SSL_CERT_FILE:-$_nel_fws_cert}"
fi
"""
    if has_gh_cli:
        setup += r"""
export GH_TOKEN="${GH_TOKEN:-fake}"
export GH_REPO="${GH_REPO:-testuser/my-project}"
# OpenClaw strips inherited GH_TOKEN/GITHUB_TOKEN from exec-tool environments;
# gh still reads this config file when routed through the fws proxy.
mkdir -p "$HOME/.config/gh"
cat > "$HOME/.config/gh/hosts.yml" <<'EOF'
github.com:
    oauth_token: fake
    git_protocol: https
EOF
"""
    return setup


def _build_openclaw_config(
    model_url: str,
    model_id: str,
    *,
    api_key: str = "",
    context_window: int | None = _DEFAULT_CONTEXT_WINDOW,
    max_tokens: int | None = None,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
    idle_timeout_seconds: int = _DEFAULT_IDLE_TIMEOUT_SECONDS,
    web_search_provider: Literal["tavily"] | None = None,
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
    }
    if context_window is not None:
        model_entry["contextWindow"] = context_window
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
        "timeoutSeconds": idle_timeout_seconds,
    }
    if api_key:
        provider_entry["apiKey"] = api_key

    tools: dict[str, Any] = {"fs": {"workspaceOnly": False}}
    plugins: dict[str, Any] = {}
    if web_search_provider:
        tools["web"] = {"search": {"provider": web_search_provider}}
        if web_search_provider == "tavily":
            plugins["entries"] = {"tavily": {"enabled": True}}

    config: dict[str, Any] = {
        "models": {
            "mode": "merge",
            "providers": {provider_name: provider_entry},
        },
        "agents": {"defaults": agent_defaults},
        "tools": tools,
    }
    if plugins:
        config["plugins"] = plugins
    return config


_WEB_SEARCH_PROVIDER_ENV_KEYS: dict[str, str] = {"tavily": "TAVILY_API_KEY"}


def _web_search_exec_env(web_search_provider: Literal["tavily"] | None) -> dict[str, str]:
    if web_search_provider == "tavily":
        api_key = os.environ.get("TAVILY_API_KEY")
        return {"TAVILY_API_KEY": api_key} if api_key else {}
    return {}


def _check_web_search_env(web_search_provider: Literal["tavily"] | None) -> None:
    """Fail fast when a web_search_provider is configured but its credential env var is missing."""
    if web_search_provider is None:
        return
    env_key = _WEB_SEARCH_PROVIDER_ENV_KEYS.get(web_search_provider)
    if env_key and not os.environ.get(env_key):
        raise RuntimeError(
            f"OpenClawSolver web_search_provider='{web_search_provider}' requires "
            f"{env_key} to be set in the environment."
        )


_WORKSPACE_SYNC_TIMEOUT_SECONDS = 120.0
_WORKSPACE_FILE_DOWNLOAD_TIMEOUT_SECONDS = 30.0
_SESSION_JSONL_FETCH_TIMEOUT_SECONDS = 30.0


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
        context_window: int | None = _DEFAULT_CONTEXT_WINDOW,
        max_tokens: int | None = None,
        max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
        idle_timeout_seconds: int = _DEFAULT_IDLE_TIMEOUT_SECONDS,
        run_timeout: float | None = None,
        timeout_strategy: str = "task",
        max_agent_timeout: float | None = None,
        web_search_provider: Literal["tavily"] | None = None,
        config_path: str | None = None,
        *,
        skip_preflight: bool = False,
    ) -> None:
        self._bin = openclaw_bin
        self._thinking = thinking
        self._timeout = timeout
        self._run_timeout = run_timeout
        self._timeout_strategy = timeout_strategy
        self._max_agent_timeout = max_agent_timeout
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._context_window = context_window
        self._max_tokens = max_tokens
        self._max_concurrent = max_concurrent
        self._idle_timeout_seconds = idle_timeout_seconds
        self._web_search_provider = web_search_provider
        self._config_path = Path(config_path) if config_path else None
        if self._config_path and not self._config_path.is_file():
            raise FileNotFoundError(f"openclaw_config points to {self._config_path} which does not exist")
        if not skip_preflight:
            _check_prerequisites(openclaw_bin)
            _check_web_search_env(web_search_provider)

    def _effective_timeout(self, task: SeedResult) -> float:
        # Resolve the agent wall-clock against the task's own ``timeout_seconds``.
        # ``timeout_strategy`` selects how (default ``task`` preserves the prior
        # min() behavior); ``override`` lets the NEL timeout win so slow-but-
        # healthy rollouts aren't SIGKILLed when upstream latency is high.
        task_timeout = _coerce_timeout(task.metadata.get("timeout_seconds"))
        if task_timeout is None:
            task_timeout = _coerce_timeout(task.metadata.get("agent_timeout_sec"))
        nel_timeout = self._run_timeout if self._run_timeout is not None else self._timeout
        if task_timeout is None or self._timeout_strategy == "override":
            effective = nel_timeout
        elif self._timeout_strategy == "max":
            effective = max(nel_timeout, task_timeout)
        else:  # "task": task budget bounded by the NEL timeout
            effective = min(nel_timeout, task_timeout)
        if self._max_agent_timeout is not None:
            effective = min(effective, self._max_agent_timeout)
        return effective

    async def solve(
        self,
        task: SeedResult,
        sandbox: Sandbox | None = None,
    ) -> SolveResult:
        if sandbox is not None:
            return await self._solve_sandbox(task, sandbox)
        if _is_pinchbench_task(task.metadata):
            err_msg = "PinchBench OpenClaw execution requires sandbox mode; local OpenClaw execution is not supported."
            return SolveResult(
                response="",
                trajectory=build_atif_trajectory(
                    [{"source": "system", "message": err_msg}],
                    agent_name="openclaw",
                    status="error",
                ),
                error=err_msg,
                error_kind=ErrorKind.INFRA,
            )
        return await self._solve_local(task)

    # ------------------------------------------------------------------
    # Sandbox path: run inside a Docker / Slurm / ECS container
    # ------------------------------------------------------------------

    async def _solve_sandbox(self, task: SeedResult, sandbox: Sandbox) -> SolveResult:
        task_id = task.metadata.get("task_id", "task")
        workspace_path = task.metadata.get("workspace_path")
        effective_timeout = self._effective_timeout(task)

        logger.info("OpenClawSolver[sandbox]: task=%s timeout=%ss", task_id, effective_timeout)

        resolved_model_url = sandbox.resolved_endpoint_url("MODEL_BASE_URL") or (
            sandbox.resolve_outside_endpoint(self._model_url) if self._model_url else self._model_url
        )
        await self._setup_config(sandbox, model_url_override=resolved_model_url)

        pre_existing_files: set[str] = set()
        if workspace_path and Path(workspace_path).is_dir():
            host_workspace = Path(workspace_path)
            await self._upload_workspace(sandbox, host_workspace)
            pre_existing_files = {
                str(p.relative_to(host_workspace)) for p in _iter_workspace_files(host_workspace) if p.is_file()
            }

        session_specs = _normalize_task_sessions(task)
        active_session_id = ""
        session_ids: list[str] = []
        effective_prompts: list[str] = []
        response_texts: list[str] = []
        envelope_steps: list[dict[str, Any]] = []
        model_responses: list[ModelResponse] = []
        oc_extra: dict[str, Any] | None = None
        agent_timeout_error: str | None = None
        raw_sessions: list[str] = []
        session_fetch_timeout = min(_SESSION_JSONL_FETCH_TIMEOUT_SECONDS, effective_timeout)
        fws_env_setup = _build_fws_env_setup(task.metadata)
        overall_deadline = time.monotonic() + effective_timeout

        await self._cleanup_sessions_sandbox(sandbox)

        for idx, session_spec in enumerate(session_specs):
            if idx == 0 or session_spec.get("new_session") or not active_session_id:
                if active_session_id:
                    raw_session_jsonl = await self._fetch_session_jsonl_sandbox(
                        sandbox,
                        active_session_id,
                        timeout_sec=session_fetch_timeout,
                    )
                    if raw_session_jsonl:
                        raw_sessions.append(raw_session_jsonl)
                    await self._cleanup_sessions_sandbox(sandbox)
                active_session_id = f"nel-{task_id}-{session_spec['id']}-{uuid.uuid4().hex[:8]}"
                session_ids.append(active_session_id)

            effective_prompt = self._build_prompt(
                str(session_spec["prompt"]),
                _CONTAINER_WORKSPACE if workspace_path else None,
            )
            effective_prompts.append(effective_prompt)

            prompt_file = f"/tmp/nel_prompt_{idx}.txt"
            await self._write_file(sandbox, prompt_file, effective_prompt)

            cmd_parts = [
                f"node {_OPENCLAW_BIN} agent",
                "--agent main",
                f'--message "$(cat {prompt_file})"',
                f"--session-id {shlex.quote(active_session_id)}",
                "--json",
                "--local",
            ]
            if self._thinking:
                cmd_parts.append(f"--thinking {shlex.quote(self._thinking)}")

            exec_env = _web_search_exec_env(self._web_search_provider)
            if self._api_key:
                exec_env["NVIDIA_API_KEY"] = self._api_key
            exec_env = exec_env or None

            turn_timeout = effective_timeout if idx == 0 else max(0.0, overall_deadline - time.monotonic())
            if turn_timeout <= 0:
                agent_timeout_error = f"OpenClaw agent timed out after {effective_timeout}s"
                envelope_steps.append({"source": "system", "message": agent_timeout_error})
                break
            agent_timeout = _format_timeout_seconds(turn_timeout)
            kill_after = _format_timeout_seconds(_OPENCLAW_AGENT_TIMEOUT_KILL_AFTER_SECONDS)
            timeout_cmd = (
                f"timeout --kill-after={shlex.quote(kill_after)} {shlex.quote(agent_timeout)} {' '.join(cmd_parts)}"
            )
            inner_cmd = (
                f"{fws_env_setup}\n"
                '_nel_timeout_marker="/tmp/nel-openclaw-timeout-$$-${RANDOM:-0}"\n'
                f'(sleep {shlex.quote(agent_timeout)}; touch "$_nel_timeout_marker") &\n'
                "_nel_timeout_watcher=$!\n"
                f"{timeout_cmd}\n"
                "rc=$?\n"
                'kill "$_nel_timeout_watcher" 2>/dev/null || true\n'
                'wait "$_nel_timeout_watcher" 2>/dev/null || true\n'
                f'if [ "$rc" -eq 124 ] || {{ [ "$rc" -eq 137 ] && [ -f "$_nel_timeout_marker" ]; }}; then echo {shlex.quote(_OPENCLAW_AGENT_TIMEOUT_SENTINEL)} >&2; fi\n'
                'rm -f "$_nel_timeout_marker"\n'
                'exit "$rc"'
            )
            full_cmd = f"bash -c {shlex.quote(inner_cmd)}"

            logger.info(
                "OpenClawSolver[sandbox]: session=%s task=%s turn=%d/%d",
                active_session_id,
                task_id,
                idx + 1,
                len(session_specs),
            )
            result = await sandbox.exec(
                full_cmd,
                timeout_sec=_sandbox_agent_exec_timeout(turn_timeout),
                env=exec_env,
            )

            if result.return_code != 0:
                stderr = _redact_secret(result.stderr, self._api_key)
                err_msg = f"exit code {result.return_code}: {stderr[:500]}"
                logger.warning("OpenClawSolver[sandbox]: %s for %s", err_msg, task_id)
                if result.return_code in (124, 137) and _OPENCLAW_AGENT_TIMEOUT_SENTINEL in result.stderr:
                    agent_timeout_error = f"OpenClaw agent timed out after {effective_timeout}s"
                    envelope_steps.append({"source": "system", "message": agent_timeout_error})
                    break
                return SolveResult(
                    response="",
                    trajectory=build_atif_trajectory(
                        [{"source": "system", "message": stderr[:2000]}],
                        agent_name="openclaw",
                        status="error",
                        extra={"exit_code": result.return_code},
                    ),
                    error=err_msg,
                    error_kind=_openclaw_error_kind(err_msg, return_code=result.return_code),
                )

            output = _extract_openclaw_envelope(result.stdout, result.stderr)
            if not output:
                stdout = _redact_secret(result.stdout, self._api_key)
                stderr = _redact_secret(result.stderr, self._api_key)
                err_msg = f"no JSON output for {task_id}"
                logger.warning("OpenClawSolver[sandbox]: %s", err_msg)
                return SolveResult(
                    response=(stdout or stderr)[:5000],
                    trajectory=build_atif_trajectory(
                        [{"source": "system", "message": stderr.strip()[:2000] or "no JSON output"}],
                        agent_name="openclaw",
                        status="error",
                    ),
                    error=err_msg,
                    error_kind=_openclaw_error_kind(f"{stdout}\n{stderr}"),
                )

            response_text, model_response, steps, turn_extra = _parse_response(output)
            response_texts.append(response_text)
            model_responses.append(model_response)
            envelope_steps.extend(steps)
            if turn_extra:
                oc_extra = turn_extra

        if workspace_path and Path(workspace_path).is_dir():
            await self._download_workspace(sandbox, Path(workspace_path), timeout_sec=effective_timeout)

        if active_session_id:
            raw_session_jsonl = await self._fetch_session_jsonl_sandbox(
                sandbox,
                active_session_id,
                timeout_sec=session_fetch_timeout,
            )
            if raw_session_jsonl:
                raw_sessions.append(raw_session_jsonl)
        raw_session_jsonl = "\n".join(raw_sessions)
        return _finalize_openclaw_solve(
            mode="sandbox",
            task_id=task_id,
            workspace_path=workspace_path,
            pre_existing_files=pre_existing_files,
            response_texts=response_texts,
            model_responses=model_responses,
            envelope_steps=envelope_steps,
            oc_extra=oc_extra,
            session_ids=session_ids,
            effective_prompts=effective_prompts,
            raw_session_jsonl=raw_session_jsonl,
            agent_timeout_error=agent_timeout_error,
            agent_timeout_seconds=effective_timeout if agent_timeout_error else None,
        )

    async def _fetch_session_jsonl_sandbox(
        self,
        sandbox: Sandbox,
        session_id: str,
        *,
        timeout_sec: float | None = None,
    ) -> str:
        """Return the OpenClaw session JSONL from the container as raw text.

        Returns an empty string when the session file is missing or
        unreadable.  The raw text is persisted verbatim to
        ``<workspace>/.nel_transcript.jsonl`` so downstream graders see
        the upstream ``{"type":"message", ...}`` envelope shape; our own
        :func:`_parse_session_jsonl` converts it into ATIF steps for the
        trajectory artifact.
        """
        operation_timeout = min(
            _SESSION_JSONL_FETCH_TIMEOUT_SECONDS,
            timeout_sec if timeout_sec is not None else _SESSION_JSONL_FETCH_TIMEOUT_SECONDS,
        )
        try:
            logger.info(
                "OpenClawSolver[sandbox]: fetching session transcript session=%s timeout=%ss",
                session_id,
                operation_timeout,
            )
            candidates: list[str] = []
            index_raw = await self._read_container_file_text(
                sandbox,
                f"{_CONTAINER_SESSIONS_DIR}/sessions.json",
                timeout_sec=operation_timeout,
            )
            if index_raw:
                candidates.extend(
                    _resolve_container_session_path(candidate)
                    for candidate in _session_path_candidates_from_index(index_raw, session_id)
                )
            candidates.extend(
                [
                    f"{_CONTAINER_SESSIONS_DIR}/{session_id}.jsonl",
                    f"{_CONTAINER_SESSIONS_DIR}/{session_id}.ndjson",
                ]
            )
            listing = await sandbox.exec(_container_session_find_command(), timeout_sec=operation_timeout)
            if listing.return_code == 0:
                candidates.extend(line.strip() for line in listing.stdout.splitlines())

            for remote_path in _unique_strings(candidates):
                raw = await self._read_container_file_text(sandbox, remote_path, timeout_sec=operation_timeout)
                if raw:
                    logger.info(
                        "OpenClawSolver[sandbox]: fetched session transcript session=%s path=%s bytes=%d",
                        session_id,
                        remote_path,
                        len(raw.encode("utf-8")),
                    )
                    return raw
            return ""
        except Exception as exc:
            logger.warning(
                "OpenClawSolver[sandbox]: failed to fetch session transcript session=%s: %s",
                session_id,
                exc,
            )
            return ""

    @staticmethod
    async def _read_container_file_text(sandbox: Sandbox, remote_path: str, *, timeout_sec: float) -> str:
        result = await sandbox.exec(
            f"base64 {shlex.quote(remote_path)} 2>/dev/null",
            timeout_sec=timeout_sec,
        )
        if result.return_code != 0 or not result.stdout.strip():
            return ""
        return _decode_b64_text(result.stdout)

    async def _cleanup_sessions_sandbox(self, sandbox: Sandbox) -> None:
        try:
            await sandbox.exec(
                f"rm -rf {shlex.quote(_CONTAINER_SESSIONS_DIR)} && mkdir -p {shlex.quote(_CONTAINER_SESSIONS_DIR)}",
                timeout_sec=5,
            )
        except Exception as exc:
            logger.debug("OpenClawSolver[sandbox]: failed to clean session directory: %s", exc)

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
                idle_timeout_seconds=self._idle_timeout_seconds,
                web_search_provider=self._web_search_provider,
            )
            config_json = json.dumps(config, indent=2)
        await self._write_file(sandbox, f"{_OPENCLAW_CONFIG_DIR}/openclaw.json", config_json)

    async def _upload_workspace(self, sandbox: Sandbox, host_workspace: Path) -> None:
        """Copy workspace files from host into the container.

        Handles both text and binary files (e.g. PDFs) correctly.
        """
        await sandbox.exec(f"mkdir -p {_CONTAINER_WORKSPACE}")
        for item in _iter_workspace_files(host_workspace):
            rel = item.relative_to(host_workspace)
            remote = f"{_CONTAINER_WORKSPACE}/{rel}"
            data = item.read_bytes()
            await self._write_file(sandbox, remote, data)

    async def _download_workspace(
        self,
        sandbox: Sandbox,
        host_workspace: Path,
        *,
        timeout_sec: float | None = None,
    ) -> None:
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

        operation_timeout = min(
            _WORKSPACE_SYNC_TIMEOUT_SECONDS,
            timeout_sec if timeout_sec is not None else _WORKSPACE_SYNC_TIMEOUT_SECONDS,
        )
        try:
            logger.info(
                "OpenClawSolver[sandbox]: listing container workspace %s timeout=%ss",
                _CONTAINER_WORKSPACE,
                operation_timeout,
            )
            listing = await sandbox.exec(_container_workspace_find_command(), timeout_sec=operation_timeout)
        except Exception as exc:
            logger.warning(
                "OpenClawSolver[sandbox]: failed to list container workspace %s: %s",
                _CONTAINER_WORKSPACE,
                exc,
            )
            return

        remote_files: list[str] = []
        skipped_files = 0
        for line in listing.stdout.splitlines():
            remote_path = line.strip()
            if not remote_path or not remote_path.startswith(_CONTAINER_WORKSPACE):
                continue
            rel = remote_path[len(_CONTAINER_WORKSPACE) + 1 :]
            if not rel:
                continue
            if _workspace_rel_is_skipped(rel):
                skipped_files += 1
                continue
            remote_files.append(remote_path)
        if not remote_files:
            logger.debug(
                "No files found in container workspace %s; skipping workspace sync",
                _CONTAINER_WORKSPACE,
            )
            return

        logger.info(
            "OpenClawSolver[sandbox]: downloading %d workspace files to %s%s",
            len(remote_files),
            host_workspace,
            f" ({skipped_files} generated/cache files skipped)" if skipped_files else "",
        )
        deadline = time.monotonic() + operation_timeout
        downloaded = 0
        for remote_path in remote_files:
            rel = remote_path[len(_CONTAINER_WORKSPACE) + 1 :]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                logger.warning(
                    "OpenClawSolver[sandbox]: workspace sync timed out after %ss; downloaded %d/%d files",
                    operation_timeout,
                    downloaded,
                    len(remote_files),
                )
                break
            local_path = host_workspace / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = await sandbox.exec(
                    f"base64 {shlex.quote(remote_path)}",
                    timeout_sec=min(_WORKSPACE_FILE_DOWNLOAD_TIMEOUT_SECONDS, remaining),
                )
                if result.return_code == 0 and result.stdout.strip():
                    data = base64.b64decode(result.stdout.strip())
                    local_path.write_bytes(data)
                    downloaded += 1
            except Exception as exc:
                logger.warning("OpenClawSolver[sandbox]: failed to download %s: %s", remote_path, exc)

        logger.info(
            "OpenClawSolver[sandbox]: downloaded %d/%d files to %s", downloaded, len(remote_files), host_workspace
        )

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
        workspace_path = task.metadata.get("workspace_path")
        effective_timeout = self._effective_timeout(task)

        pre_existing_files: set[str] = set()
        if workspace_path and Path(workspace_path).is_dir():
            host_workspace = Path(workspace_path)
            pre_existing_files = {
                str(p.relative_to(host_workspace)) for p in _iter_workspace_files(host_workspace) if p.is_file()
            }

        env = {**os.environ}
        if self._api_key:
            env["NVIDIA_API_KEY"] = self._api_key
        _tmp_home = self._prepare_local_config(env)
        cwd = workspace_path if workspace_path and Path(workspace_path).is_dir() else None

        session_specs = _normalize_task_sessions(task)
        active_session_id = ""
        session_ids: list[str] = []
        effective_prompts: list[str] = []
        response_texts: list[str] = []
        envelope_steps: list[dict[str, Any]] = []
        model_responses: list[ModelResponse] = []
        oc_extra: dict[str, Any] | None = None
        agent_timeout_error: str | None = None
        raw_sessions: list[str] = []
        overall_deadline = time.monotonic() + effective_timeout
        solve_started_at = time.time()

        for idx, session_spec in enumerate(session_specs):
            if idx == 0 or session_spec.get("new_session") or not active_session_id:
                if active_session_id:
                    raw_session_jsonl = self._fetch_session_jsonl_local(
                        env.get("HOME", str(Path.home())),
                        active_session_id,
                        started_at=solve_started_at,
                    )
                    if raw_session_jsonl:
                        raw_sessions.append(raw_session_jsonl)
                active_session_id = f"nel-{task_id}-{session_spec['id']}-{uuid.uuid4().hex[:8]}"
                session_ids.append(active_session_id)

            effective_prompt = self._build_prompt(str(session_spec["prompt"]), workspace_path)
            effective_prompts.append(effective_prompt)

            cmd = [
                self._bin,
                "agent",
                "--message",
                effective_prompt,
                "--session-id",
                active_session_id,
                "--json",
                "--local",
            ]
            if self._thinking:
                cmd.extend(["--thinking", self._thinking])

            logger.info(
                "OpenClawSolver[local]: session=%s task=%s turn=%d/%d workspace=%s",
                active_session_id,
                task_id,
                idx + 1,
                len(session_specs),
                cwd or "(none)",
            )

            turn_timeout = effective_timeout if idx == 0 else max(0.0, overall_deadline - time.monotonic())
            if turn_timeout <= 0:
                agent_timeout_error = f"Timeout after {effective_timeout}s"
                envelope_steps.append({"source": "system", "message": agent_timeout_error})
                break

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
                    timeout=turn_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                err_msg = f"Timeout after {effective_timeout}s"
                logger.warning("OpenClawSolver[local]: %s for %s", err_msg, task_id)
                agent_timeout_error = err_msg
                envelope_steps.append({"source": "system", "message": err_msg})
                break

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
                    error_kind=_openclaw_error_kind(err_msg, return_code=process.returncode),
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
                    error_kind=_openclaw_error_kind(f"{stdout}\n{stderr}"),
                )

            response_text, model_response, steps, turn_extra = _parse_response(output)
            response_texts.append(response_text)
            model_responses.append(model_response)
            envelope_steps.extend(steps)
            if turn_extra:
                oc_extra = turn_extra

        home = env.get("HOME", str(Path.home()))
        if active_session_id:
            raw_session_jsonl = self._fetch_session_jsonl_local(home, active_session_id, started_at=solve_started_at)
            if raw_session_jsonl:
                raw_sessions.append(raw_session_jsonl)
        raw_session_jsonl = "\n".join(raw_sessions)
        return _finalize_openclaw_solve(
            mode="local",
            task_id=task_id,
            workspace_path=workspace_path,
            pre_existing_files=pre_existing_files,
            response_texts=response_texts,
            model_responses=model_responses,
            envelope_steps=envelope_steps,
            oc_extra=oc_extra,
            session_ids=session_ids,
            effective_prompts=effective_prompts,
            raw_session_jsonl=raw_session_jsonl,
            agent_timeout_error=agent_timeout_error,
            agent_timeout_seconds=effective_timeout if agent_timeout_error else None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fetch_session_jsonl_local(home: str, session_id: str, *, started_at: float | None = None) -> str:
        """Return a local session transcript as raw JSONL text, or empty."""
        sessions_dir = Path(home) / ".openclaw" / "agents" / "main" / "sessions"
        candidates: list[Path] = []
        index_path = sessions_dir / "sessions.json"
        if index_path.exists():
            raw_index = index_path.read_text(encoding="utf-8", errors="replace")
            candidates.extend(
                _resolve_local_session_path(sessions_dir, candidate)
                for candidate in _session_path_candidates_from_index(raw_index, session_id)
            )
        candidates.extend([sessions_dir / f"{session_id}.jsonl", sessions_dir / f"{session_id}.ndjson"])
        if sessions_dir.exists():
            min_mtime = None if started_at is None else max(0.0, started_at - 5.0)
            candidates.extend(
                sorted(
                    [
                        path
                        for path in [*sessions_dir.glob("*.jsonl"), *sessions_dir.glob("*.ndjson")]
                        if min_mtime is None or path.stat().st_mtime >= min_mtime
                    ],
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
        seen: set[Path] = set()
        for transcript_path in candidates:
            if transcript_path in seen:
                continue
            seen.add(transcript_path)
            if transcript_path.exists():
                return transcript_path.read_text(encoding="utf-8", errors="replace")
        return ""

    @staticmethod
    def _build_prompt(raw_prompt: str, workspace_path: str | None) -> str:
        _ = workspace_path
        return raw_prompt

    async def close(self) -> None:
        pass
