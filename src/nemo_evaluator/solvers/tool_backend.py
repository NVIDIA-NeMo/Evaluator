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
"""Pluggable tool backends for the NEL-driven agent loop (ReActSolver).

Three implementations:

- ``HttpToolBackend``: Gym Resource Server tools via HTTP (GET /openapi.json,
  POST /{tool}).
- ``SandboxToolBackend``: Sandbox-local tools (bash, file_read, file_write,
  str_replace_editor) via ``Sandbox.exec()`` / ``upload()`` / ``download()``.
- ``CompositeToolBackend``: Merges multiple backends, routes by tool name.

All backends conform to the ``ToolBackend`` protocol so ``ReActSolver`` is
backend-agnostic.
"""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from dataclasses import dataclass, field

from nemo_evaluator.errors import GracefulError
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import aiohttp

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_CWD_MARKER = "__NEL_CWD__"


# ── Data types ────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Structured return from a tool call."""

    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolInfraError(GracefulError):
    """Unrecoverable tool infrastructure failure — scores 0.0, no retry."""


# ── Protocol ──────────────────────────────────────────────────────────


class ToolBackend(Protocol):
    async def list_tools(self) -> list[dict[str, Any]]: ...
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult: ...
    async def close(self) -> None: ...


# ── HttpToolBackend ───────────────────────────────────────────────────


class HttpToolBackend:
    """Gym Resource Server tools.  Discovers via ``/openapi.json``,
    dispatches via ``POST /{tool_name}``."""

    def __init__(
        self,
        base_url: str,
        *,
        session_id: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._session_id = session_id
        self._timeout = timeout
        self._session: aiohttp.ClientSession | None = None
        self._tools_cache: list[dict[str, Any]] | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._tools_cache is not None:
            return self._tools_cache

        session = self._get_session()
        try:
            async with session.get(f"{self._base_url}/openapi.json") as resp:
                resp.raise_for_status()
                spec = await resp.json()
        except Exception as exc:
            raise ToolInfraError(f"Failed to fetch OpenAPI spec from {self._base_url}: {exc}") from exc

        tools = _openapi_to_tools(spec)
        self._tools_cache = tools
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        session = self._get_session()
        url = f"{self._base_url}/{name}"
        body: dict[str, Any] = dict(arguments)
        if self._session_id:
            body.setdefault("session_id", self._session_id)

        try:
            async with session.post(url, json=body) as resp:
                text = await resp.text()
                if resp.status >= 400:
                    return ToolResult(
                        content=f"HTTP {resp.status}: {text[:4096]}",
                        is_error=True,
                        metadata={"http_status": resp.status},
                    )
                try:
                    data = json.loads(text)
                    content = json.dumps(data, indent=2) if isinstance(data, (dict, list)) else text
                except json.JSONDecodeError:
                    content = text
                return ToolResult(content=content[:16384])
        except asyncio.TimeoutError:
            return ToolResult(content=f"Tool {name!r} timed out after {self._timeout}s", is_error=True)
        except aiohttp.ClientError as exc:
            raise ToolInfraError(f"HTTP call to {url} failed: {exc}") from exc

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


# ── SandboxToolBackend ────────────────────────────────────────────────


_SANDBOX_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_read",
            "description": "Read contents of a file from the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_write",
            "description": "Write content to a file in the sandbox (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to the file."},
                    "content": {"type": "string", "description": "Content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": (
                "Make a targeted replacement in a file. "
                "Use command='view' to read, 'str_replace' to replace, 'create' to create."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace"],
                        "description": "The operation to perform.",
                    },
                    "path": {"type": "string", "description": "Absolute path to the file."},
                    "old_str": {"type": "string", "description": "String to find (for str_replace)."},
                    "new_str": {"type": "string", "description": "Replacement string (for str_replace/create)."},
                    "view_range": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Optional [start_line, end_line] for view.",
                    },
                },
                "required": ["command", "path"],
            },
        },
    },
]


class SandboxToolBackend:
    """Sandbox tools: bash, file_read, file_write, str_replace_editor.

    Constructed per-problem inside ``ReActSolver.solve()`` with a live
    ``Sandbox`` instance.  File operations use ``upload()`` / ``download()``
    to avoid shell-escaping user content.
    """

    def __init__(self, sandbox: Sandbox, *, timeout: float = 180.0) -> None:
        self._sandbox = sandbox
        self._timeout = timeout
        self._cwd: str = sandbox.spec.workdir if sandbox.spec else "/workspace"
        self._tmp_files: list[Path] = []

    async def list_tools(self) -> list[dict[str, Any]]:
        return list(_SANDBOX_TOOL_SCHEMAS)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        dispatch = {
            "bash": self._bash,
            "file_read": self._file_read,
            "file_write": self._file_write,
            "str_replace_editor": self._str_replace_editor,
        }
        handler = dispatch.get(name)
        if handler is None:
            raise ToolInfraError(f"Unknown sandbox tool: {name!r}")
        try:
            return await handler(arguments)
        except ToolInfraError:
            raise
        except Exception as exc:
            logger.warning("Sandbox tool %s failed: %s", name, exc, exc_info=True)
            return ToolResult(content=str(exc), is_error=True)

    # ── bash ──────────────────────────────────────────────────────────

    async def _bash(self, args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        if not command:
            return ToolResult(content="No command provided.", is_error=True)

        cmd_with_cwd = f"{command} ; echo '{_CWD_MARKER}='$(pwd)"
        try:
            result = await self._sandbox.exec(
                cmd_with_cwd,
                timeout_sec=self._timeout,
                cwd=self._cwd,
            )
        except Exception as exc:
            raise ToolInfraError(f"Sandbox exec failed: {exc}") from exc

        stdout = result.stdout or ""
        self._update_cwd(stdout)

        output_parts = []
        if stdout:
            clean = self._strip_cwd_marker(stdout)
            if clean:
                output_parts.append(clean)
        if result.stderr:
            output_parts.append(f"STDERR:\n{result.stderr}")

        content = "\n".join(output_parts) or "(no output)"
        if result.return_code != 0:
            content = f"[exit code {result.return_code}]\n{content}"

        return ToolResult(
            content=content,
            is_error=result.return_code != 0,
            metadata={"exit_code": result.return_code},
        )

    def _update_cwd(self, stdout: str) -> None:
        marker = f"{_CWD_MARKER}="
        for line in reversed(stdout.splitlines()):
            if line.startswith(marker):
                new_cwd = line[len(marker) :].strip()
                if new_cwd:
                    self._cwd = new_cwd
                return

    def _strip_cwd_marker(self, stdout: str) -> str:
        lines = stdout.splitlines()
        marker = f"{_CWD_MARKER}="
        cleaned = [ln for ln in lines if not ln.startswith(marker)]
        return "\n".join(cleaned).rstrip()

    # ── file_read ─────────────────────────────────────────────────────

    async def _file_read(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        if not path:
            return ToolResult(content="No path provided.", is_error=True)

        tmp = Path(tempfile.mktemp(suffix=".download"))
        self._tmp_files.append(tmp)
        try:
            await self._sandbox.download(path, tmp)
            content = tmp.read_text(errors="replace")
        except Exception as exc:
            return ToolResult(content=f"Failed to read {path}: {exc}", is_error=True)
        return ToolResult(content=content)

    # ── file_write ────────────────────────────────────────────────────

    async def _file_write(self, args: dict[str, Any]) -> ToolResult:
        path = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return ToolResult(content="No path provided.", is_error=True)

        tmp = Path(tempfile.mktemp(suffix=".upload"))
        self._tmp_files.append(tmp)
        try:
            tmp.write_text(content)
            parent = str(Path(path).parent)
            await self._sandbox.exec(f"mkdir -p {parent}", timeout_sec=10, cwd=self._cwd)
            await self._sandbox.upload(tmp, path)
        except Exception as exc:
            return ToolResult(content=f"Failed to write {path}: {exc}", is_error=True)
        return ToolResult(content=f"Written {len(content)} bytes to {path}.")

    # ── str_replace_editor ────────────────────────────────────────────

    async def _str_replace_editor(self, args: dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        path = args.get("path", "")
        if not path:
            return ToolResult(content="No path provided.", is_error=True)

        if command == "view":
            return await self._editor_view(path, args.get("view_range"))
        elif command == "create":
            new_str = args.get("new_str", "")
            return await self._file_write({"path": path, "content": new_str})
        elif command == "str_replace":
            return await self._editor_replace(path, args.get("old_str", ""), args.get("new_str", ""))
        else:
            return ToolResult(content=f"Unknown command: {command!r}", is_error=True)

    async def _editor_view(self, path: str, view_range: list[int] | None = None) -> ToolResult:
        result = await self._file_read({"path": path})
        if result.is_error:
            return result
        lines = result.content.splitlines()
        if view_range and len(view_range) == 2:
            start, end = max(1, view_range[0]), min(len(lines), view_range[1])
            lines = lines[start - 1 : end]
            numbered = [f"{start + i:6d}\t{ln}" for i, ln in enumerate(lines)]
        else:
            numbered = [f"{i + 1:6d}\t{ln}" for i, ln in enumerate(lines)]
        return ToolResult(content="\n".join(numbered))

    async def _editor_replace(self, path: str, old_str: str, new_str: str) -> ToolResult:
        if not old_str:
            return ToolResult(content="old_str is required for str_replace.", is_error=True)

        read_result = await self._file_read({"path": path})
        if read_result.is_error:
            return read_result

        content = read_result.content
        count = content.count(old_str)
        if count == 0:
            return ToolResult(content=f"old_str not found in {path}.", is_error=True)
        if count > 1:
            return ToolResult(
                content=f"old_str found {count} times in {path}. Must be unique.",
                is_error=True,
            )

        new_content = content.replace(old_str, new_str, 1)
        return await self._file_write({"path": path, "content": new_content})

    async def close(self) -> None:
        for tmp in self._tmp_files:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
        self._tmp_files.clear()


# ── CompositeToolBackend ──────────────────────────────────────────────


class CompositeToolBackend:
    """Merges multiple backends, routes ``call_tool`` by tool name."""

    def __init__(self, backends: list[ToolBackend]) -> None:
        self._backends = backends
        self._route: dict[str, ToolBackend] = {}

    async def list_tools(self) -> list[dict[str, Any]]:
        all_tools: list[dict[str, Any]] = []
        self._route.clear()
        for backend in self._backends:
            tools = await backend.list_tools()
            for tool in tools:
                name = tool.get("function", {}).get("name", "")
                if name and name not in self._route:
                    self._route[name] = backend
                    all_tools.append(tool)
        return all_tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        backend = self._route.get(name)
        if backend is None:
            raise ToolInfraError(f"Tool {name!r} not found in any backend. Available: {sorted(self._route.keys())}")
        return await backend.call_tool(name, arguments)

    async def close(self) -> None:
        for backend in self._backends:
            try:
                await backend.close()
            except Exception:
                logger.debug("Backend close failed", exc_info=True)


# ── OpenAPI → OpenAI tool schema conversion ───────────────────────────


def _openapi_to_tools(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert an OpenAPI 3.x spec into OpenAI function-calling tool schemas.

    Filters to POST endpoints only (Gym tool endpoints). Skips internal
    endpoints (``/seed_session``, ``/verify``, ``/run``, ``/health``, etc.).
    """
    _SKIP_PATHS = {"/seed_session", "/verify", "/run", "/health", "/openapi.json", "/docs", "/redoc"}
    tools: list[dict[str, Any]] = []
    paths = spec.get("paths", {})
    components_schemas = spec.get("components", {}).get("schemas", {})

    for path, methods in paths.items():
        if path in _SKIP_PATHS:
            continue
        post = methods.get("post")
        if post is None:
            continue

        name = path.lstrip("/").replace("/", "_")
        description = post.get("summary") or post.get("description") or name

        parameters = _extract_parameters(post, components_schemas)

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )

    return tools


def _extract_parameters(
    operation: dict[str, Any],
    schemas: dict[str, Any],
) -> dict[str, Any]:
    """Extract JSON Schema parameters from an OpenAPI POST operation."""
    body = operation.get("requestBody", {})
    content = body.get("content", {})
    json_schema = content.get("application/json", {}).get("schema", {})

    if "$ref" in json_schema:
        ref_name = json_schema["$ref"].split("/")[-1]
        json_schema = schemas.get(ref_name, {})

    if not json_schema:
        return {"type": "object", "properties": {}}

    resolved = _resolve_refs(json_schema, schemas)
    resolved.pop("title", None)
    return resolved


def _resolve_refs(schema: dict[str, Any], schemas: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve $ref in a JSON schema."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        resolved = dict(schemas.get(ref_name, {}))
        resolved.pop("title", None)
        return _resolve_refs(resolved, schemas)

    result = dict(schema)
    if "properties" in result:
        result["properties"] = {k: _resolve_refs(v, schemas) for k, v in result["properties"].items()}
    if "items" in result:
        result["items"] = _resolve_refs(result["items"], schemas)
    for key in ("allOf", "anyOf", "oneOf"):
        if key in result:
            result[key] = [_resolve_refs(s, schemas) for s in result[key]]
    return result
