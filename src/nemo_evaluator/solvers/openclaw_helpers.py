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
"""Pure helpers for OpenClaw solver output, workspaces, and transcripts."""

from __future__ import annotations

import base64
import json
import logging
import os
import shlex
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

from nemo_evaluator.observability.types import ModelResponse

from .base import ErrorKind, SolveResult
from .trajectory_util import build_atif_trajectory

logger = logging.getLogger(__name__)

_OPENCLAW_CONFIG_DIR = "/home/node/.openclaw"
_CONTAINER_WORKSPACE = "/home/node/workspace"
_CONTAINER_SESSIONS_DIR = f"{_OPENCLAW_CONFIG_DIR}/agents/main/sessions"

_SANDBOX_AGENT_TIMEOUT_GRACE_SECONDS = 30.0

_RETRYABLE_OPENCLAW_ERROR_FRAGMENTS = (
    "reasoning-only assistant turn detected",
    "MidStreamFallbackError",
    "APIConnectionError",
    "OpenAIException - list index out of range",
    "ECONNRESET",
    "ETIMEDOUT",
    "ECONNREFUSED",
    "socket hang up",
    "upstream request timeout",
)
_RETRYABLE_OPENCLAW_AGENT_RESPONSE_PREFIX = "⚠️ Agent couldn't generate a response"

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
_WORKSPACE_SKIP_DIR_NAMES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".openclaw",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "test-results",
        "venv",
    }
)
_TRANSCRIPT_FILENAME = ".nel_transcript.jsonl"
_MAX_SCAN_BYTES = 4_000_000
_JSON_DECODER = json.JSONDecoder()


def _merge_model_responses(responses: list[ModelResponse], *, content: str) -> ModelResponse:
    """Aggregate provider-reported usage from multiple OpenClaw turns."""
    if not responses:
        return ModelResponse(content=content)
    last = responses[-1]

    def _sum(attr: str) -> int | None:
        values = [getattr(response, attr) for response in responses]
        present = [int(value) for value in values if value is not None]
        if not present:
            return None
        return sum(present)

    return ModelResponse(
        content=content,
        model=last.model,
        finish_reason=last.finish_reason,
        prompt_tokens=_sum("prompt_tokens"),
        completion_tokens=_sum("completion_tokens"),
        total_tokens=_sum("total_tokens"),
        reasoning_tokens=_sum("reasoning_tokens"),
        latency_ms=sum(response.latency_ms for response in responses),
        raw_response=last.raw_response,
        request_prompt=last.request_prompt,
        request_system=last.request_system,
    )


def _finalize_openclaw_solve(
    *,
    mode: str,
    task_id: str,
    workspace_path: str | None,
    pre_existing_files: set[str],
    response_texts: list[str],
    model_responses: list[ModelResponse],
    envelope_steps: list[dict[str, Any]],
    oc_extra: dict[str, Any] | None,
    session_ids: list[str],
    effective_prompts: list[str],
    raw_session_jsonl: str,
    agent_timeout_error: str | None = None,
    agent_timeout_seconds: float | None = None,
) -> SolveResult:
    response_text = "\n\n".join(text for text in response_texts if text)
    model_response = _merge_model_responses(model_responses, content=response_text)

    if workspace_path:
        file_addendum = _read_workspace_files(Path(workspace_path), pre_existing_files)
        if file_addendum:
            response_text = f"{response_text}\n\n{file_addendum}" if response_text else file_addendum
            model_response.content = response_text
    else:
        file_addendum = ""

    steps = envelope_steps
    if raw_session_jsonl:
        transcript_steps = _parse_session_jsonl(raw_session_jsonl)
        if transcript_steps:
            # Envelope `payloads` are openclaw's user-facing rendering of the
            # final assistant message; with the session transcript available,
            # plain-text payloads duplicate `transcript_steps[-1]` and inflate
            # `agent_steps` past the real wire model-call count. Keep only
            # envelope steps that carry unique extras (error or media payloads).
            envelope_steps = [s for s in envelope_steps if s.get("extra")]
            steps = transcript_steps + envelope_steps
        if workspace_path:
            _persist_session_transcript(Path(workspace_path), raw_session_jsonl)

    if not response_text.strip() and agent_timeout_error:
        response_text = agent_timeout_error
        model_response.content = response_text

    extra = dict(oc_extra or {})
    if agent_timeout_error:
        extra["openclaw_timed_out"] = True
        extra["openclaw_timeout_seconds"] = agent_timeout_seconds
    if len(session_ids) > 1:
        extra["openclaw_session_ids"] = session_ids

    trajectory = build_atif_trajectory(
        steps,
        agent_name="openclaw",
        model_name=model_response.model or None,
        prompt_tokens=model_response.prompt_tokens,
        completion_tokens=model_response.completion_tokens,
        extra=extra,
        user_prompt="\n\n".join(effective_prompts),
    )

    logger.info(
        "OpenClawSolver[%s]: %s completed in %sms, %s tok, %d steps",
        mode,
        task_id,
        model_response.latency_ms,
        model_response.total_tokens,
        len(steps),
    )
    retryable_error = _openclaw_retryable_response_error(
        response_text,
        has_file_addendum=bool(file_addendum),
    )
    if retryable_error is not None:
        err_msg, err_kind = retryable_error
        return SolveResult(
            response=response_text,
            model_response=model_response,
            trajectory=trajectory,
            error=err_msg,
            error_kind=err_kind,
        )
    return SolveResult(
        response=response_text,
        model_response=model_response,
        trajectory=trajectory,
    )


def _format_timeout_seconds(timeout: float) -> str:
    value = float(timeout)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:.3f}".rstrip("0").rstrip(".") + "s"


def _sandbox_agent_exec_timeout(agent_timeout: float) -> float:
    return agent_timeout + _SANDBOX_AGENT_TIMEOUT_GRACE_SECONDS


def _redact_secret(text: str, secret: str | None) -> str:
    if not secret:
        return text
    return text.replace(secret, "[REDACTED]")


def _coerce_timeout(value: Any) -> float | None:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    return timeout if timeout > 0 else None


def _workspace_rel_is_skipped(rel: str | PurePosixPath) -> bool:
    return any(part in _WORKSPACE_SKIP_DIR_NAMES for part in PurePosixPath(str(rel)).parts)


def _iter_workspace_files(workspace: Path) -> Iterator[Path]:
    if not workspace.is_dir():
        return
    for root, dirnames, filenames in os.walk(workspace):
        dirnames[:] = sorted(dirname for dirname in dirnames if dirname not in _WORKSPACE_SKIP_DIR_NAMES)
        root_path = Path(root)
        for filename in sorted(filenames):
            yield root_path / filename


def _container_workspace_find_command() -> str:
    skip_expr = " -o ".join(f"-name {shlex.quote(name)}" for name in sorted(_WORKSPACE_SKIP_DIR_NAMES))
    return (
        f"find {shlex.quote(_CONTAINER_WORKSPACE)} "
        f"-type d \\( {skip_expr} \\) -prune -o -type f -print 2>/dev/null || true"
    )


def _read_workspace_files(
    workspace: Path,
    pre_existing: set[str] | None = None,
) -> str:
    """Read *new* text files from the workspace and format them for the response."""
    if pre_existing is None:
        pre_existing = set()

    parts: list[str] = []
    for path in _iter_workspace_files(workspace):
        rel = path.relative_to(workspace)
        rel_str = str(rel)
        if _workspace_rel_is_skipped(rel_str):
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


def _session_path_candidates_from_index(raw: str, session_id: str) -> list[str]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    candidates: list[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str):
            return
        stripped = value.strip()
        if stripped.endswith((".jsonl", ".ndjson")) or session_id in stripped:
            candidates.append(stripped)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key in ("path", "file", "filename", "sessionFile", "jsonl", "jsonlPath"):
                add(value.get(key))
            if any(str(value.get(key, "")) == session_id for key in ("id", "sessionId", "session_id", "uuid")):
                for nested in value.values():
                    add(nested)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)
        else:
            add(value)

    visit(data)
    return _unique_strings(candidates)


def _resolve_container_session_path(candidate: str) -> str:
    path = PurePosixPath(candidate)
    if path.is_absolute():
        return str(path)
    return str(PurePosixPath(_CONTAINER_SESSIONS_DIR) / path)


def _resolve_local_session_path(sessions_dir: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return sessions_dir / path


def _unique_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            unique.append(value)
    return unique


def _container_session_find_command() -> str:
    return (
        f"find {shlex.quote(_CONTAINER_SESSIONS_DIR)} "
        "\\( -name '*.jsonl' -o -name '*.ndjson' \\) -type f "
        "-printf '%T@ %p\\n' 2>/dev/null | sort -nr | cut -d' ' -f2- || true"
    )


def _decode_b64_text(raw: str) -> str:
    return base64.b64decode(raw.strip()).decode("utf-8", errors="replace")


def _persist_session_transcript(workspace: Path, raw_jsonl: str) -> None:
    """Write OpenClaw's session JSONL verbatim to ``<workspace>/.nel_transcript.jsonl``."""
    try:
        if not workspace.is_dir():
            return
        (workspace / _TRANSCRIPT_FILENAME).write_text(raw_jsonl, encoding="utf-8")
    except OSError as exc:
        logger.warning("Failed to persist session transcript to %s: %s", workspace, exc)


def _iter_json_objects(raw: str) -> Iterator[dict[str, Any]]:
    """Yield every top-level JSON object embedded in ``raw``, in order."""
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
    """Locate the ``openclaw agent --json`` result envelope."""
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


def _openclaw_error_kind(message: str, *, return_code: int | None = None) -> ErrorKind:
    if return_code in (124, 137, -1) or "timed out after" in message:
        return ErrorKind.INFRA
    if any(fragment in message for fragment in _RETRYABLE_OPENCLAW_ERROR_FRAGMENTS):
        return ErrorKind.INFRA
    return ErrorKind.GRACEFUL


def _openclaw_retryable_response_error(
    response_text: str, *, has_file_addendum: bool = False
) -> tuple[str, ErrorKind] | None:
    text = response_text.strip()
    if not text:
        if has_file_addendum:
            return None
        return "OpenClaw returned an empty visible response", ErrorKind.INFRA
    if any(fragment in text for fragment in _RETRYABLE_OPENCLAW_ERROR_FRAGMENTS):
        return text.splitlines()[0][:500], ErrorKind.INFRA
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(_RETRYABLE_OPENCLAW_AGENT_RESPONSE_PREFIX):
            return line[:500], ErrorKind.INFRA
    return None


def _parse_response(output: dict[str, Any]) -> tuple[str, ModelResponse, list[dict[str, Any]], dict[str, Any] | None]:
    """Extract response text, model response, and trajectory from openclaw JSON output."""
    payloads = output.get("payloads") or []
    response_text = "\n".join(p.get("text", "").strip() for p in payloads if p.get("text"))
    meta = output.get("meta", {})
    duration_ms = meta.get("durationMs", 0)
    agent_meta = meta.get("agentMeta", {})

    usage = meta.get("usage") if isinstance(meta.get("usage"), dict) else {}
    prompt_tokens = _usage_int(usage, "inputTokens", "prompt_tokens")
    completion_tokens = _usage_int(usage, "outputTokens", "completion_tokens")
    total_tokens = _usage_int(usage, "totalTokens", "total_tokens")
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    model_response = ModelResponse(
        content=response_text,
        model=agent_meta.get("model", ""),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        latency_ms=float(duration_ms),
    )

    steps: list[dict[str, Any]] = []
    payload_errors: list[str] = []
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
            if msg_text:
                payload_errors.append(msg_text)
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
    if payload_errors:
        if oc_extra is None:
            oc_extra = {}
        oc_extra["payload_errors"] = payload_errors

    return response_text, model_response, steps, oc_extra


def _usage_int(usage: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key in usage and usage.get(key) is not None:
            try:
                return int(usage[key])
            except (TypeError, ValueError):
                return None
    return None


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
        if record.get("type") == "message" and isinstance(record.get("message"), dict):
            record = record["message"]
        role = record.get("role", "")
        if role == "assistant":
            content = record.get("content", [])
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []
            reasoning_parts: list[str] = []
            for block in content if isinstance(content, list) else []:
                btype = block.get("type", "")
                if btype == "text" and block.get("text"):
                    text_parts.append(block["text"])
                elif btype == "thinking" and block.get("thinking"):
                    reasoning_parts.append(block["thinking"])
                elif btype == "redacted_thinking":
                    reasoning_parts.append("[redacted_thinking]")
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
            if reasoning_parts:
                step["reasoning_content"] = "\n".join(reasoning_parts)
            if tool_calls:
                step["tool_calls"] = tool_calls
            if text_parts or tool_calls or reasoning_parts:
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
