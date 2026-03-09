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


def _resolve_image(image: str) -> str:
    """Convert an image reference to a format the OpenAI vision API accepts.

    Handles: http(s) URLs (pass through), data URIs (pass through),
    local file paths (convert to base64 data URI).
    """
    if image.startswith(("http://", "https://", "data:")):
        return image

    import base64
    import mimetypes
    from pathlib import Path

    p = Path(image)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {image}")

    mime = mimetypes.guess_type(image)[0] or "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


@dataclass
class ModelClient:
    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 2048
    top_p: float | None = None
    seed: int | None = None
    timeout: float = 120.0
    max_concurrent: int = 8
    cache_dir: str | None = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    reasoning_pattern: str | None = None
    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _http: httpx.AsyncClient | None = field(init=False, repr=False, default=None)
    _cache: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent)
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        if self.cache_dir:
            from nemo_evaluator.runner.cache import ResponseCache
            self._cache = ResponseCache(self.cache_dir)

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

    async def chat(self, prompt: str | None = None, system: str | None = None,
                   messages: list[dict[str, str]] | None = None) -> ModelResponse:
        if messages is not None:
            msgs = list(messages)
        else:
            msgs = []
            if system:
                msgs.append({"role": "system", "content": system})
            msgs.append({"role": "user", "content": prompt or ""})

        cache_prompt = prompt or (msgs[-1]["content"] if msgs else "")
        cache_msgs = messages if messages is not None else None

        if self._cache:
            cached = self._cache.get(
                self.model, cache_prompt, system,
                self.temperature, self.max_tokens,
                top_p=self.top_p, seed=self.seed, messages=cache_msgs,
            )
            if cached:
                return self._parse_response(cached, 0.0, cache_prompt, system)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.seed is not None:
            payload["seed"] = self.seed

        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        data = await self._post_with_retry(url, payload)
        latency = (time.monotonic() - t0) * 1000

        if self._cache:
            self._cache.put(
                self.model, cache_prompt, system,
                self.temperature, self.max_tokens, data,
                top_p=self.top_p, seed=self.seed, messages=cache_msgs,
            )
        return self._parse_response(data, latency, cache_prompt, system)

    async def vlm_chat(
        self,
        prompt: str,
        images: list[str],
        system: str | None = None,
        detail: str = "auto",
    ) -> ModelResponse:
        """Multimodal chat with interleaved text + image content blocks.

        Images can be URLs (http/https), local file paths, or base64 data URIs.
        Follows the OpenAI vision API format.
        """
        content: list[dict[str, Any]] = []
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": _resolve_image(img), "detail": detail}})
        content.append({"type": "text", "text": prompt})

        msgs: list[dict[str, Any]] = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.seed is not None:
            payload["seed"] = self.seed

        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        data = await self._post_with_retry(url, payload)
        latency = (time.monotonic() - t0) * 1000

        if self._cache:
            self._cache.put(self.model, prompt, system,
                            self.temperature, self.max_tokens, data,
                            top_p=self.top_p, seed=self.seed)
        return self._parse_response(data, latency, prompt, system)

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict:
        """Shared retry logic for all HTTP endpoints."""
        last_exc: Exception | None = None

        for attempt in range(self.retry.max_retries + 1):
            async with self._sem:
                try:
                    client = self._get_client()
                    resp = await client.post(url, json=payload)

                    if resp.status_code in self.retry.retry_on_status and attempt < self.retry.max_retries:
                        delay = self._backoff_delay(attempt)
                        logger.warning("Retryable status %d on attempt %d/%d, sleeping %.1fs",
                                       resp.status_code, attempt + 1, self.retry.max_retries + 1, delay)
                        await asyncio.sleep(delay)
                        continue

                    resp.raise_for_status()
                    return resp.json()

                except httpx.TimeoutException as e:
                    last_exc = e
                    if attempt < self.retry.max_retries:
                        await asyncio.sleep(self._backoff_delay(attempt))
                        continue
                    raise
                except httpx.HTTPStatusError:
                    raise
                except Exception as e:
                    last_exc = e
                    if attempt < self.retry.max_retries:
                        await asyncio.sleep(self._backoff_delay(attempt))
                        continue
                    raise

        raise last_exc or RuntimeError("All retries exhausted")

    async def embed(self, text: str) -> list[float]:
        """Get embedding for a single text via /v1/embeddings."""
        result = await self.embed_batch([text])
        return result[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts via /v1/embeddings."""
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        data = await self._post_with_retry(url, payload)
        embeddings_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in embeddings_data]

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

        content = choice["message"].get("content") or ""
        if self.reasoning_pattern:
            import re as _re
            content = _re.sub(self.reasoning_pattern, "", content, flags=_re.DOTALL).strip()

        return ModelResponse(
            content=content,
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
