from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from nemo_evaluator.models import RetryConfig
from nemo_evaluator.observability.types import ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class ModelClient:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: float = 120.0
    max_concurrent: int = 8
    retry: RetryConfig = field(default_factory=RetryConfig)
    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _http: httpx.AsyncClient | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent)
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get_client(self) -> httpx.AsyncClient:
        if self._http is None or self._http.is_closed:
            self._http = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=self.max_concurrent * 2,
                    max_keepalive_connections=self.max_concurrent,
                ),
                headers=self._headers(),
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.is_closed:
            await self._http.aclose()
            self._http = None

    async def chat(self, prompt: str, system: str | None = None) -> ModelResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        url = f"{self.base_url}/chat/completions"
        last_exc: Exception | None = None

        for attempt in range(self.retry.max_retries + 1):
            async with self._sem:
                t0 = time.monotonic()
                try:
                    client = self._get_client()
                    resp = await client.post(url, json=payload)

                    if resp.status_code in self.retry.retry_on_status and attempt < self.retry.max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning(
                            "Retryable status %d on attempt %d/%d, sleeping %.1fs",
                            resp.status_code, attempt + 1, self.retry.max_retries + 1, delay,
                        )
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    latency = (time.monotonic() - t0) * 1000

                except httpx.TimeoutException as e:
                    last_exc = e
                    if attempt < self.retry.max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning("Timeout on attempt %d/%d, retrying in %.1fs",
                                       attempt + 1, self.retry.max_retries + 1, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise

                except httpx.HTTPStatusError:
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt < self.retry.max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning("Error on attempt %d/%d: %s, retrying in %.1fs",
                                       attempt + 1, self.retry.max_retries + 1, e, delay)
                        await asyncio.sleep(delay)
                        continue
                    raise

            return self._parse_response(data, latency, prompt, system)

        raise last_exc or RuntimeError("All retries exhausted")

    def _backoff_delay(self, attempt: int) -> float:
        delay = min(self.retry.base_delay * (2 ** attempt), self.retry.max_delay)
        if self.retry.jitter:
            delay *= 0.5 + random.random()
        return delay

    def _parse_response(self, data: dict, latency: float,
                        prompt: str, system: str | None) -> ModelResponse:
        choices = data.get("choices", [])
        if not choices:
            raise ValueError(f"No choices in response: {data}")

        choice = choices[0]
        usage = data.get("usage", {})

        reasoning_tokens = 0
        ct = usage.get("completion_tokens_details") or {}
        reasoning_tokens = ct.get("reasoning_tokens", 0)

        return ModelResponse(
            content=choice["message"]["content"],
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            reasoning_tokens=reasoning_tokens,
            latency_ms=round(latency, 2),
            raw_response=data,
            request_prompt=prompt,
            request_system=system,
        )
