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
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult

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
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    max_concurrent: int = _DEFAULT_MAX_CONCURRENT,
) -> dict[str, Any]:
    """Build an OpenClaw config for an arbitrary OpenAI-compatible provider.

    The provider name is derived from the ``model_id`` prefix
    (e.g. ``nvidia`` from ``nvidia/nemotron-...``).
    """
    provider_name = model_id.split("/")[0] if "/" in model_id else "custom"
    openclaw_model_ref = f"{provider_name}/{model_id}"

    return {
        "models": {
            "mode": "merge",
            "providers": {
                provider_name: {
                    "baseUrl": model_url,
                    "apiKey": api_key or "NVIDIA_API_KEY",
                    "api": "openai-completions",
                    "models": [
                        {
                            "id": model_id,
                            "name": model_id,
                            "input": ["text"],
                            "contextWindow": context_window,
                            "maxTokens": max_tokens,
                        },
                    ],
                },
            },
        },
        "agents": {
            "defaults": {
                "model": {"primary": openclaw_model_ref},
                "workspace": _CONTAINER_WORKSPACE,
                "compaction": {"mode": "safeguard"},
                "maxConcurrent": max_concurrent,
            },
        },
        "tools": {"fs": {"workspaceOnly": False}},
    }


_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml",
    ".csv", ".html", ".css", ".ics", ".xml", ".toml", ".cfg", ".ini",
    ".sh", ".bash", ".log", ".rst", ".tex",
})
_MAX_FILE_SIZE = 50_000


def _read_workspace_files(workspace: Path) -> str:
    """Read text files from the workspace and format them for the response.

    Agent solvers create files that automated graders check directly.
    But the LLM-as-judge pipeline only sees ``SolveResult.response``.
    Including file contents in the response lets the judge evaluate
    the actual deliverable, not just the agent's summary.
    """
    parts: list[str] = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _TEXT_EXTENSIONS:
            continue
        if path.stat().st_size > _MAX_FILE_SIZE:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace").rstrip()
        except OSError:
            continue
        rel = path.relative_to(workspace)
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


def _parse_response(output: dict[str, Any]) -> tuple[str, ModelResponse, list[dict[str, Any]]]:
    """Extract response text, model response, and trajectory from openclaw JSON output."""
    payloads = output.get("payloads") or []
    response_text = "\n".join(
        p.get("text", "").strip()
        for p in payloads
        if p.get("text")
    )
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

    trajectory: list[dict[str, Any]] = []
    for p in payloads:
        entry: dict[str, Any] = {
            "type": "message",
            "message": {"role": "assistant", "content": []},
        }
        if p.get("text"):
            entry["message"]["content"].append({"type": "text", "text": p["text"]})
        if p.get("mediaUrl"):
            entry["message"]["content"].append({"type": "media", "url": p["mediaUrl"]})
        if p.get("mediaUrls"):
            for url in p["mediaUrls"]:
                entry["message"]["content"].append({"type": "media", "url": url})
        if p.get("isError"):
            entry["is_error"] = True
        if entry["message"]["content"]:
            trajectory.append(entry)

    if agent_meta:
        trajectory.append({
            "type": "openclaw_meta",
            "provider": agent_meta.get("provider", ""),
            "model": agent_meta.get("model", ""),
            "duration_ms": duration_ms,
            "session_id": agent_meta.get("sessionId", ""),
        })

    return response_text, model_response, trajectory


def _parse_session_jsonl(raw: str) -> list[dict[str, Any]]:
    """Parse an OpenClaw session JSONL transcript into trajectory entries.

    Extracts assistant messages (text + tool_use blocks) and tool results
    to give evaluators full visibility into what the agent did.
    """
    entries: list[dict[str, Any]] = []
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
            msg: dict[str, Any] = {
                "type": "message",
                "message": {"role": "assistant", "content": []},
            }
            for block in (content if isinstance(content, list) else []):
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    msg["message"]["content"].append({"type": "text", "text": block["text"]})
                elif btype == "tool_use":
                    msg["message"]["content"].append({
                        "type": "tool_use",
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    })
            if msg["message"]["content"]:
                entries.append(msg)
        elif role == "tool":
            content = record.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            entries.append({
                "type": "tool_result",
                "tool_use_id": record.get("tool_use_id", ""),
                "content": str(content)[:2000],
            })
    return entries


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
        max_tokens: int = _DEFAULT_MAX_TOKENS,
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
            raise FileNotFoundError(
                f"openclaw_config points to {self._config_path} which does not exist"
            )
        if not skip_preflight:
            _check_prerequisites(openclaw_bin)

    async def solve(
        self, task: SeedResult, sandbox: Sandbox | None = None,
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

        await self._setup_config(sandbox)

        if workspace_path and Path(workspace_path).is_dir():
            await self._upload_workspace(sandbox, Path(workspace_path))

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
            logger.warning(
                "OpenClawSolver[sandbox]: exit code %d for %s: %s",
                result.return_code, task_id, result.stderr[:500],
            )
            return SolveResult(
                response="",
                trajectory=[{
                    "error": "openclaw_exit",
                    "code": result.return_code,
                    "stderr": result.stderr[:2000],
                }],
            )

        output = _extract_json(result.stdout)
        if not output:
            logger.warning("OpenClawSolver[sandbox]: no JSON in stdout for %s", task_id)
            return SolveResult(
                response=result.stdout[:5000],
                trajectory=[{"warning": "no_json_output", "stderr": result.stderr[:2000]}],
            )

        response_text, model_response, trajectory = _parse_response(output)

        if workspace_path:
            file_addendum = _read_workspace_files(Path(workspace_path))
            if file_addendum:
                response_text = f"{response_text}\n\n{file_addendum}" if response_text else file_addendum

        transcript = await self._read_session_transcript(sandbox, session_id)
        if transcript:
            trajectory = transcript + trajectory

        logger.info(
            "OpenClawSolver[sandbox]: %s completed in %dms, %d tok, %d trajectory entries",
            task_id, model_response.latency_ms, model_response.total_tokens, len(trajectory),
        )
        return SolveResult(
            response=response_text,
            model_response=model_response,
            trajectory=trajectory,
        )

    async def _read_session_transcript(
        self, sandbox: Sandbox, session_id: str,
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

    async def _setup_config(self, sandbox: Sandbox) -> None:
        """Write an OpenClaw config into the container.

        Uses ``config_path`` verbatim when provided, otherwise generates
        one from the solver parameters.
        """
        if self._config_path is not None:
            config_json = self._config_path.read_text(encoding="utf-8")
        else:
            config = _build_openclaw_config(
                model_url=self._model_url,
                model_id=self._model_id,
                api_key=self._api_key or "",
                context_window=self._context_window,
                max_tokens=self._max_tokens,
                max_concurrent=self._max_concurrent,
            )
            config_json = json.dumps(config, indent=2)
        await self._write_file(sandbox, f"{_OPENCLAW_CONFIG_DIR}/openclaw.json", config_json)

    async def _upload_workspace(self, sandbox: Sandbox, host_workspace: Path) -> None:
        """Copy workspace files from host into the container."""
        await sandbox.exec(f"mkdir -p {_CONTAINER_WORKSPACE}")
        for item in host_workspace.rglob("*"):
            if item.is_file():
                rel = item.relative_to(host_workspace)
                remote = f"{_CONTAINER_WORKSPACE}/{rel}"
                content = item.read_text(encoding="utf-8", errors="replace")
                await self._write_file(sandbox, remote, content)

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

        listing = await sandbox.exec(
            f"find {_CONTAINER_WORKSPACE} -type f 2>/dev/null || true"
        )
        remote_files = [
            line.strip() for line in listing.stdout.splitlines()
            if line.strip() and line.strip().startswith(_CONTAINER_WORKSPACE)
        ]
        if not remote_files:
            logger.warning(
                "No files found in container workspace %s; agent may have written elsewhere",
                _CONTAINER_WORKSPACE,
            )
        for remote_path in remote_files:
            rel = remote_path[len(_CONTAINER_WORKSPACE) + 1:]
            if not rel:
                continue
            local_path = host_workspace / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                result = await sandbox.exec(
                    f"base64 {shlex.quote(remote_path)}"
                )
                if result.return_code == 0 and result.stdout.strip():
                    data = base64.b64decode(result.stdout.strip())
                    local_path.write_bytes(data)
            except Exception as exc:
                logger.debug("Failed to download %s: %s", remote_path, exc)

        logger.debug("Downloaded %d files to %s", len(remote_files), host_workspace)

    @staticmethod
    async def _write_file(sandbox: Sandbox, remote_path: str, content: str) -> None:
        """Write a file inside the container via base64-encoded exec.

        Files are created by the container user (avoids ``docker cp``
        ownership issues with the non-root ``node`` user).
        """
        import base64

        b64 = base64.b64encode(content.encode()).decode()
        remote_dir = str(Path(remote_path).parent)
        await sandbox.exec(f"mkdir -p {shlex.quote(remote_dir)}")
        await sandbox.exec(f"echo {shlex.quote(b64)} | base64 -d > {shlex.quote(remote_path)}")

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

        effective_prompt = self._build_prompt(task.prompt, workspace_path)

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
                process.communicate(), timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("OpenClawSolver[local]: timeout after %.0fs for %s", self._timeout, task_id)
            return SolveResult(
                response="",
                trajectory=[{"error": "openclaw_timeout", "timeout": self._timeout}],
            )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            logger.warning(
                "OpenClawSolver[local]: exit code %d for %s: %s",
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
            logger.warning("OpenClawSolver[local]: no JSON in stdout for %s", task_id)
            return SolveResult(
                response=stdout[:5000],
                trajectory=[{"warning": "no_json_output", "stderr": stderr[:2000]}],
            )

        response_text, model_response, trajectory = _parse_response(output)

        if workspace_path:
            file_addendum = _read_workspace_files(Path(workspace_path))
            if file_addendum:
                response_text = f"{response_text}\n\n{file_addendum}" if response_text else file_addendum

        home = env.get("HOME", str(Path.home()))
        transcript = self._read_local_session_transcript(home, session_id)
        if transcript:
            trajectory = transcript + trajectory

        logger.info(
            "OpenClawSolver[local]: %s completed in %dms, %d tok, %d trajectory entries",
            task_id, model_response.latency_ms, model_response.total_tokens, len(trajectory),
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
        home: str, session_id: str,
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
