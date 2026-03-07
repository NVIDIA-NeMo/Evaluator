"""Solver protocol: pluggable inference strategy for the eval loop."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class SolveResult:
    response: str
    model_response: ModelResponse | None = None
    trajectory: list[dict[str, Any]] = field(default_factory=list)


class Solver(Protocol):
    async def solve(self, task: SeedResult) -> SolveResult: ...


class ChatSolver:
    """Single model call. Default for standard benchmarks."""

    def __init__(self, client: Any, system_prompt: str | None = None) -> None:
        self._client = client
        self._system = system_prompt

    async def solve(self, task: SeedResult) -> SolveResult:
        effective_system = self._system or task.system
        if task.messages:
            resp = await self._client.chat(messages=task.messages)
        else:
            resp = await self._client.chat(task.prompt, system=effective_system)
        return SolveResult(response=resp.content, model_response=resp)

    async def close(self) -> None:
        await self._client.close()


class CompletionSolver:
    """Uses the /completions endpoint instead of /chat/completions."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 2048) -> None:
        import httpx
        self._url = f"{base_url.rstrip('/')}/completions"
        self._model = model
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = httpx.AsyncClient(timeout=120.0)

    async def solve(self, task: SeedResult) -> SolveResult:
        import time
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        t0 = time.monotonic()
        resp = await self._client.post(self._url, json={
            "model": self._model,
            "prompt": task.prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        latency = (time.monotonic() - t0) * 1000

        text = data["choices"][0].get("text", "")
        usage = data.get("usage", {})
        model_resp = ModelResponse(
            content=text, model=data.get("model", self._model),
            finish_reason=data["choices"][0].get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=round(latency, 2),
            raw_response=data,
        )
        return SolveResult(response=text, model_response=model_resp)

    async def close(self) -> None:
        await self._client.aclose()


class AgentSolver:
    """Wraps an external agent framework (OpenHands, SWE-agent, etc.).

    The agent receives the task prompt and produces a solution through
    multi-turn interaction with tools/environments. NEL doesn't own
    the inner loop -- the agent does.
    """

    def __init__(
        self,
        agent_cmd: str,
        model_url: str,
        model_id: str,
        api_key: str | None = None,
        max_turns: int = 100,
        timeout: float = 1800.0,
    ) -> None:
        self._agent_cmd = agent_cmd
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._max_turns = max_turns
        self._timeout = timeout

    async def solve(self, task: SeedResult) -> SolveResult:
        import asyncio
        import json
        import shlex
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(prefix="nel_agent_") as tmpdir:
            task_file = Path(tmpdir) / "task.json"
            task_file.write_text(json.dumps({
                "prompt": task.prompt,
                "expected_answer": task.expected_answer,
                "metadata": task.metadata,
            }))

            output_file = Path(tmpdir) / "output.json"

            cmd = (
                f"{self._agent_cmd} "
                f"--task-file {task_file} "
                f"--output-file {output_file} "
                f"--model-url {self._model_url} "
                f"--model-id {self._model_id} "
                f"--max-turns {self._max_turns}"
            )

            import os
            env = {**os.environ}
            if self._api_key:
                env["NEL_API_KEY"] = self._api_key

            process = await asyncio.create_subprocess_shell(
                f"/bin/bash -c {shlex.quote(cmd)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SolveResult(response="", trajectory=[{"error": "agent_timeout"}])

            if output_file.exists():
                result = json.loads(output_file.read_text())
                return SolveResult(
                    response=result.get("response", ""),
                    trajectory=result.get("trajectory", []),
                )

            return SolveResult(
                response=stdout.decode()[:5000] if stdout else "",
                trajectory=[{"stderr": stderr.decode()[:2000] if stderr else ""}],
            )

    async def close(self) -> None:
        pass
