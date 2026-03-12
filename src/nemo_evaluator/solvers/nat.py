"""NatSolver: general-purpose solver that delegates to a NAT agent via HTTP SSE.

Works with ANY benchmark -- sends the task prompt to a running ``nat serve``
instance and returns the final text response.  When the task metadata includes
``workspace_path``, the full trajectory is additionally written as JSONL so
that workspace-aware scorers (e.g. PinchBench) can inspect tool calls.

The NAT server must already be running (``nat serve --config_file ...``).

SSE protocol (``/generate/full``):
  - ``intermediate_data: <json>`` -- raw IntermediateStep events
  - ``data: <json>``             -- final result payload
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 600.0


@dataclass
class _SSEEvent:
    kind: str  # "intermediate_data" | "data" | "observability_trace"
    payload: dict[str, Any] = field(default_factory=dict)


def _parse_sse_lines(raw: str) -> list[_SSEEvent]:
    """Parse NAT's SSE format into structured events.

    NAT emits lines like ``intermediate_data: {JSON}`` and ``data: {JSON}``.
    The ``payload`` field inside ``intermediate_data`` events is double-encoded
    (a JSON string inside a JSON object) and must be decoded twice.
    """
    events: list[_SSEEvent] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith(":"):
            continue
        for prefix in ("intermediate_data:", "data:", "observability_trace:"):
            if line.startswith(prefix):
                body = line[len(prefix):].strip()
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    logger.debug("SSE parse skip: %s", body[:120])
                    break

                if prefix == "intermediate_data:" and isinstance(obj.get("payload"), str):
                    try:
                        obj["payload"] = json.loads(obj["payload"])
                    except json.JSONDecodeError:
                        pass

                events.append(_SSEEvent(kind=prefix.rstrip(":"), payload=obj))
                break
    return events


def _convert_trajectory(events: list[_SSEEvent]) -> list[dict[str, Any]]:
    """Convert NAT IntermediateStep events into the OpenClaw-style transcript
    format that PinchBench grade() functions expect.

    Mapping:
      LLM_END   -> assistant message with text content
      TOOL_START -> toolCall content appended to preceding assistant message
      TOOL_END   -> toolResult message
    """
    transcript: list[dict[str, Any]] = []

    for ev in events:
        if ev.kind != "intermediate_data":
            continue
        p = ev.payload
        step_type = p.get("type", "")

        if isinstance(step_type, dict):
            step_type = step_type.get("type", str(step_type))

        payload_data = p.get("payload", {})
        if isinstance(payload_data, dict):
            data = payload_data.get("data", {})
        else:
            data = {}

        if step_type == "LLM_END":
            output_text = ""
            if isinstance(data, dict):
                output_text = str(data.get("output", data.get("chunk", "")))
            elif isinstance(data, str):
                output_text = data
            transcript.append({
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}],
                },
            })

        elif step_type == "TOOL_START":
            tool_name = p.get("name", "unknown_tool")
            tool_input = data.get("input", "") if isinstance(data, dict) else str(data)
            tool_call = {
                "type": "toolCall",
                "name": tool_name,
                "arguments": tool_input if isinstance(tool_input, dict) else {"raw": str(tool_input)},
            }
            if transcript and transcript[-1].get("message", {}).get("role") == "assistant":
                transcript[-1]["message"]["content"].append(tool_call)
            else:
                transcript.append({
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "content": [tool_call],
                    },
                })

        elif step_type == "TOOL_END":
            tool_output = ""
            if isinstance(data, dict):
                tool_output = str(data.get("output", ""))
            elif isinstance(data, str):
                tool_output = data
            transcript.append({
                "type": "message",
                "message": {
                    "role": "toolResult",
                    "content": [tool_output],
                },
            })

    return transcript


class NatSolver:
    """Solver that communicates with a NAT agent via its ``/v1/workflow/full``
    SSE endpoint.

    General-purpose: works with any benchmark.  The prompt is sent as
    ``input_message``; the final ``data:`` event's ``output`` field becomes
    the response text.

    Accepts an optional ``sandbox`` parameter for forward compatibility and
    API consistency (the eval loop checks for this via introspection), but
    does not currently interact with the sandbox.
    """

    def __init__(
        self,
        nat_url: str = "http://localhost:8000",
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._nat_url = nat_url.rstrip("/")
        self._timeout = timeout
        self._health_checked = False

    async def _ensure_health(self) -> None:
        if self._health_checked:
            return
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(f"{self._nat_url}/health")
                r.raise_for_status()
                logger.info("NAT server healthy at %s", self._nat_url)
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as exc:
            logger.warning("NAT health check failed (%s) -- will attempt solve anyway", exc)
        self._health_checked = True

    async def solve(
        self, task: SeedResult, sandbox: Sandbox | None = None,
    ) -> SolveResult:
        await self._ensure_health()

        url = f"{self._nat_url}/generate/full"
        payload = {"input_message": task.prompt}

        t0 = time.monotonic()
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            raw_text = resp.text
        latency = (time.monotonic() - t0) * 1000

        events = _parse_sse_lines(raw_text)

        final_text = ""
        for ev in reversed(events):
            if ev.kind == "data":
                p = ev.payload
                final_text = str(p.get("output", p.get("value", p.get("response", ""))))
                break

        if not final_text:
            for ev in reversed(events):
                if ev.kind == "intermediate_data":
                    step_type = ev.payload.get("type", "")
                    if step_type == "LLM_END":
                        data = ev.payload.get("payload", {})
                        if isinstance(data, dict):
                            data = data.get("data", {})
                        if isinstance(data, dict):
                            final_text = str(data.get("output", ""))
                        break

        trajectory = _convert_trajectory(events)

        workspace_path = task.metadata.get("workspace_path")
        if workspace_path:
            transcript_file = Path(workspace_path) / ".nel_transcript.jsonl"
            try:
                transcript_file.parent.mkdir(parents=True, exist_ok=True)
                with open(transcript_file, "w", encoding="utf-8") as f:
                    for entry in trajectory:
                        f.write(json.dumps(entry) + "\n")
                logger.debug("Wrote %d transcript entries to %s", len(trajectory), transcript_file)
            except OSError as exc:
                logger.warning("Failed to write transcript: %s", exc)

        total_tokens = len(final_text) // 4 if final_text else 0
        model_resp = ModelResponse(
            content=final_text,
            model="nat-agent",
            total_tokens=total_tokens,
            completion_tokens=total_tokens,
            latency_ms=round(latency, 2),
        )

        logger.info("NatSolver: %.0fms, %d events, %d chars response",
                     latency, len(events), len(final_text))

        return SolveResult(
            response=final_text,
            model_response=model_resp,
            trajectory=trajectory,
        )

    async def close(self) -> None:
        pass
