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
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from nemo_evaluator.errors import InfraError
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.schemas import RetryConfig

logger = logging.getLogger(__name__)


@dataclass
class ToolCallInfo:
    """A single parsed tool call from a model response."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolCallingResponse:
    """Typed return from ``ModelClient.chat_with_tools()``."""

    content: str
    tool_calls: list[ToolCallInfo]
    finish_reason: str
    model_response: ModelResponse
    reasoning_content: str = ""


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
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    seed: int | None = None
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    timeout: float = 120.0
    max_concurrent: int = 8
    cache_dir: str | None = None
    retry: RetryConfig = field(default_factory=RetryConfig)
    reasoning_pattern: str | None = None
    _sem: asyncio.Semaphore = field(init=False, repr=False)
    _http: aiohttp.ClientSession | None = field(init=False, repr=False, default=None)
    _cache: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.max_concurrent)
        self.base_url = self.base_url.rstrip("/")
        if self.base_url.endswith("/chat/completions"):
            self.base_url = self.base_url[: -len("/chat/completions")]
        if self.cache_dir:
            from nemo_evaluator.engine.cache import ResponseCache

            self._cache = ResponseCache(self.cache_dir)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _get_client(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(
                    limit=self.max_concurrent * 2,
                    limit_per_host=self.max_concurrent,
                ),
                headers=self._headers(),
            )
        return self._http

    async def close(self) -> None:
        if self._http and not self._http.closed:
            await self._http.close()
            self._http = None

    _GENERATION_KEYS = ("temperature", "max_tokens", "top_p", "seed", "stop", "frequency_penalty", "presence_penalty")

    def _build_generation_payload(self, **overrides: Any) -> dict[str, Any]:
        """Build generation params dict, with optional per-call overrides.

        For each known generation key, the override wins if present,
        otherwise the client default is used.  ``None`` values are omitted.
        Any extra override keys are passed through as-is.
        """
        params: dict[str, Any] = {}
        for key in self._GENERATION_KEYS:
            val = overrides.pop(key, getattr(self, key))
            if val is not None:
                params[key] = val
        for k, v in overrides.items():
            params[k] = v
        return params

    async def chat(
        self, prompt: str | None = None, system: str | None = None, messages: list[dict[str, str]] | None = None
    ) -> ModelResponse:
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
                self.model,
                cache_prompt,
                system,
                self.temperature,
                self.max_tokens,
                top_p=self.top_p,
                seed=self.seed,
                stop=self.stop,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                messages=cache_msgs,
            )
            if cached:
                return self._parse_response(cached, 0.0, cache_prompt, system)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": msgs,
            **self._build_generation_payload(),
        }

        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        data = await self._post_with_retry(url, payload)
        latency = (time.monotonic() - t0) * 1000

        if self._cache:
            self._cache.put(
                self.model,
                cache_prompt,
                system,
                self.temperature,
                self.max_tokens,
                data,
                top_p=self.top_p,
                seed=self.seed,
                stop=self.stop,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                messages=cache_msgs,
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
            **self._build_generation_payload(),
        }

        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        data = await self._post_with_retry(url, payload)
        latency = (time.monotonic() - t0) * 1000

        if self._cache:
            self._cache.put(
                self.model,
                prompt,
                system,
                self.temperature,
                self.max_tokens,
                data,
                top_p=self.top_p,
                seed=self.seed,
                stop=self.stop,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
            )
        return self._parse_response(data, latency, prompt, system)

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        **overrides: Any,
    ) -> ToolCallingResponse:
        """Tool-augmented chat call.  Returns typed ``ToolCallingResponse``
        with parsed tool calls.  ``overrides`` can set ``temperature``,
        ``max_tokens``, etc."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            **self._build_generation_payload(**overrides),
        }

        url = f"{self.base_url}/chat/completions"
        t0 = time.monotonic()
        data = await self._post_with_retry(url, payload)
        latency = (time.monotonic() - t0) * 1000

        choices = data.get("choices", [])
        if not choices:
            raise InfraError(
                "Model returned HTTP 200 but empty choices in tool-calling response. "
                "Possible KV-cache exhaustion or model crash."
            )

        choice = choices[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
        finish_reason = choice.get("finish_reason", "")

        tool_calls: list[ToolCallInfo] = []
        for raw_tc in message.get("tool_calls") or []:
            fn = raw_tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            try:
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                parsed_args = {"raw": raw_args}
            tool_calls.append(
                ToolCallInfo(
                    id=raw_tc.get("id", ""),
                    name=fn.get("name", ""),
                    arguments=parsed_args,
                )
            )

        ct = usage.get("completion_tokens_details") or {}
        model_response = ModelResponse(
            content=content,
            model=data.get("model", self.model),
            finish_reason=finish_reason,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=ct.get("reasoning_tokens"),
            latency_ms=round(latency, 2),
            raw_response=data,
            request_prompt=None,
            request_system=None,
        )

        return ToolCallingResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            model_response=model_response,
            reasoning_content=reasoning_content,
        )

    async def _post_with_retry(self, url: str, payload: dict[str, Any]) -> dict:
        """Shared retry logic for all HTTP endpoints."""
        last_exc: Exception | None = None

        for attempt in range(self.retry.max_retries + 1):
            async with self._sem:
                try:
                    client = self._get_client()
                    async with client.post(url, json=payload) as resp:
                        if resp.status in self.retry.retry_on_status and attempt < self.retry.max_retries:
                            delay = self._backoff_delay(attempt)
                            body = await resp.text()
                            logger.warning(
                                "Retryable status %d on attempt %d/%d, sleeping %.1fs | body: %.500s",
                                resp.status,
                                attempt + 1,
                                self.retry.max_retries + 1,
                                delay,
                                body,
                            )
                            await asyncio.sleep(delay)
                            continue

                        resp.raise_for_status()
                        return await resp.json()

                except asyncio.TimeoutError as e:
                    last_exc = e
                    if attempt < self.retry.max_retries:
                        await asyncio.sleep(self._backoff_delay(attempt))
                        continue
                    raise InfraError(f"Model request timed out after {self.retry.max_retries} retries") from e
                except aiohttp.ClientResponseError as e:
                    raise InfraError(f"Model returned HTTP {e.status} after retries: {e.message}") from e
                except Exception as e:
                    last_exc = e
                    if attempt < self.retry.max_retries:
                        await asyncio.sleep(self._backoff_delay(attempt))
                        continue
                    raise InfraError(f"Model endpoint unreachable after {self.retry.max_retries} retries: {e}") from e

        raise last_exc or InfraError("All retries exhausted")

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
        delay = min(self.retry.base_delay * (2**attempt), self.retry.max_delay)
        if self.retry.jitter:
            delay *= 0.5 + random.random()
        return delay

    def _parse_response(self, data: dict, latency: float, prompt: str, system: str | None) -> ModelResponse:
        choices = data.get("choices", [])
        if not choices:
            raise InfraError("Model returned HTTP 200 but empty choices. Possible KV-cache exhaustion or model crash.")

        choice = choices[0]
        usage = data.get("usage", {})

        ct = usage.get("completion_tokens_details") or {}

        content = choice["message"].get("content") or ""
        if self.reasoning_pattern:
            import re as _re

            content = _re.sub(self.reasoning_pattern, "", content, flags=_re.DOTALL).strip()

        return ModelResponse(
            content=content,
            model=data.get("model", self.model),
            finish_reason=choice.get("finish_reason", ""),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            reasoning_tokens=ct.get("reasoning_tokens"),
            latency_ms=round(latency, 2),
            raw_response=data,
            request_prompt=prompt,
            request_system=system,
        )
