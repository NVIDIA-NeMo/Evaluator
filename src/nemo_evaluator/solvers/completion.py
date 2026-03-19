"""CompletionSolver: uses the /completions endpoint instead of /chat/completions."""
from __future__ import annotations

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.observability.types import ModelResponse

from .base import SolveResult
from .trajectory_util import build_single_turn_atif


class CompletionSolver:
    """Uses the /completions endpoint instead of /chat/completions."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 2048) -> None:
        from nemo_evaluator.runner.model_client import ModelClient
        self._model_client = ModelClient(
            base_url=base_url.rstrip("/"), model=model, api_key=api_key,
            temperature=temperature, max_tokens=max_tokens,
        )
        self._url = f"{base_url.rstrip('/')}/completions"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def solve(self, task: SeedResult) -> SolveResult:
        import time
        payload = {
            "model": self._model,
            "prompt": task.prompt,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        t0 = time.monotonic()
        data = await self._model_client._post_with_retry(self._url, payload)
        latency = (time.monotonic() - t0) * 1000

        text = data["choices"][0].get("text", "")
        usage = data.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        model_resp = ModelResponse(
            content=text, model=data.get("model", self._model),
            finish_reason=data["choices"][0].get("finish_reason", ""),
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=usage.get("total_tokens", 0),
            latency_ms=round(latency, 2),
            raw_response=data,
        )
        trajectory = build_single_turn_atif(
            task.prompt, text,
            model_name=self._model,
            prompt_tokens=pt,
            completion_tokens=ct,
        )
        return SolveResult(response=text, model_response=model_resp, trajectory=trajectory)

    async def close(self) -> None:
        await self._model_client.close()
