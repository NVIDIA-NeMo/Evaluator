# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NeMo Evaluator client with integrated adapter support for client-mode evaluation."""

import asyncio
import os
import weakref
from typing import Any, List, Optional

from openai import AsyncOpenAI
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm_asyncio

from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
from nemo_evaluator.client.adapter_transport import create_async_adapter_http_client
from nemo_evaluator.logging import get_logger

logger = get_logger(__name__)


class NeMoEvaluatorClient:
    """Async OpenAI-compatible client with integrated adapter pipeline support.

    This client enables "client mode" for adapters, where the adapter interceptor
    pipeline runs in-process through the HTTP client, instead of requiring a
    separate proxy server.

    Args:
        endpoint_model_config: Endpoint model configuration
        output_dir: Directory for adapter output files
    """

    def __init__(
        self,
        endpoint_model_config: EndpointModelConfig,
        output_dir: str,
    ):
        self.model_id = endpoint_model_config.model_id
        self.model_url = endpoint_model_config.url
        self.api_key_name = endpoint_model_config.api_key_name
        self.adapter_config = endpoint_model_config.adapter_config
        self.temperature = endpoint_model_config.temperature
        self.top_p = endpoint_model_config.top_p
        self.max_new_tokens = endpoint_model_config.max_new_tokens
        self.request_timeout = endpoint_model_config.request_timeout
        self.max_retries = endpoint_model_config.max_retries or 5
        self.parallelism = endpoint_model_config.parallelism or 1
        self.output_dir = output_dir

        is_base_url = endpoint_model_config.is_base_url or False

        adapter_http_client, self.adapter_transport = create_async_adapter_http_client(
            endpoint_url=self.model_url,
            adapter_config=self.adapter_config,
            output_dir=self.output_dir,
            is_base_url=is_base_url,
            model_name=self.model_id,
        )

        self.client = AsyncOpenAI(
            http_client=adapter_http_client,
            base_url=self.model_url,
            api_key=os.getenv(self.api_key_name, "dummy_api_key")
            if self.api_key_name is not None
            else "dummy_api_key",
            timeout=self.request_timeout,
            max_retries=0,  # We handle retries ourselves
        )

        self.semaphore = asyncio.Semaphore(self.parallelism)

        # Register finalizer to ensure post-eval hooks run even if user forgets to close
        # This is a safety net - using context manager (async with) is still recommended
        def cleanup_hooks(transport_ref):
            """Run post-eval hooks on object destruction if not already run."""
            transport = transport_ref()
            if transport is not None:
                try:
                    transport.run_post_eval_hooks()
                    logger.info("Post-eval hooks executed via finalizer")
                except Exception as e:
                    logger.error(f"Failed to run post-eval hooks in finalizer: {e}")

        self._finalizer = weakref.finalize(
            self, cleanup_hooks, weakref.ref(self.adapter_transport)
        )

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff."""
        attempt = 0
        async for attempt_state in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=2, min=2, max=120),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        ):
            with attempt_state:
                attempt = attempt_state.retry_state.attempt_number
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < self.max_retries:
                        logger.warning(
                            f"Request failed (attempt {attempt}/{self.max_retries}): {type(e).__name__}: {str(e)}"
                        )
                    raise

    async def __call__(
        self, messages: list[dict[str, Any]], seed: int | None = None
    ) -> str:
        return await self.chat_completion(messages, seed=seed)

    async def chat_completion(
        self, messages: list[dict[str, Any]], seed: Optional[int] = None
    ) -> str:
        params = {
            "model": self.model_id,
            "messages": messages,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            params["max_tokens"] = self.max_new_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if seed is not None:
            params["seed"] = seed

        async def _make_request():
            async with self.semaphore:
                return await self.client.chat.completions.create(**params)

        response = await self._retry_with_backoff(_make_request)
        response_text = response.choices[0].message.content
        return response_text or ""

    def chat_completions(
        self,
        messages_list: List[list[dict[str, Any]]],
        seeds: Optional[List[Optional[int]]] = None,
        show_progress: bool = True,
    ) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "chat_completions cannot be called from within an async context. "
                    "Use async batch_chat_completions instead."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.batch_chat_completions(messages_list, seeds, show_progress)
            )
        finally:
            pass

    async def batch_chat_completions(
        self,
        messages_list: List[list[dict[str, Any]]],
        seeds: Optional[List[Optional[int]]] = None,
        show_progress: bool = True,
    ) -> List[str]:
        if seeds is None:
            seeds = [None] * len(messages_list)
        elif len(seeds) != len(messages_list):
            raise ValueError(
                f"Length of seeds ({len(seeds)}) must match length of messages_list ({len(messages_list)})"
            )

        tasks = [
            self.chat_completion(messages, seed=seed)
            for messages, seed in zip(messages_list, seeds)
        ]

        if show_progress:
            results = await tqdm_asyncio.gather(*tasks, desc="Chat completions")
        else:
            results = await asyncio.gather(*tasks)

        return results

    async def completion(
        self, prompt: str, seed: Optional[int] = None, **kwargs
    ) -> str:
        params = {
            "model": self.model_id,
            "prompt": prompt,
            **kwargs,
        }

        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            params["max_tokens"] = self.max_new_tokens
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if seed is not None:
            params["seed"] = seed

        async def _make_request():
            async with self.semaphore:
                return await self.client.completions.create(**params)

        response = await self._retry_with_backoff(_make_request)
        return response.choices[0].text or ""

    def completions(
        self,
        prompts: List[str],
        seeds: Optional[List[Optional[int]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[str]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "completions cannot be called from within an async context. "
                    "Use async batch_completions instead."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.batch_completions(prompts, seeds, show_progress, **kwargs)
            )
        finally:
            pass

    async def batch_completions(
        self,
        prompts: List[str],
        seeds: Optional[List[Optional[int]]] = None,
        show_progress: bool = True,
        **kwargs,
    ) -> List[str]:
        if seeds is None:
            seeds = [None] * len(prompts)
        elif len(seeds) != len(prompts):
            raise ValueError(
                f"Length of seeds ({len(seeds)}) must match length of prompts ({len(prompts)})"
            )

        tasks = [
            self.completion(prompt, seed=seed, **kwargs)
            for prompt, seed in zip(prompts, seeds)
        ]

        if show_progress:
            results = await tqdm_asyncio.gather(*tasks, desc="Completions")
        else:
            results = await asyncio.gather(*tasks)

        return results

    async def embedding(self, text: str, **kwargs) -> List[float]:
        params = {
            "model": self.model_id,
            "input": text,
            **kwargs,
        }

        async def _make_request():
            async with self.semaphore:
                return await self.client.embeddings.create(**params)

        response = await self._retry_with_backoff(_make_request)
        return response.data[0].embedding

    def embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
        **kwargs,
    ) -> List[List[float]]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError(
                    "embeddings cannot be called from within an async context. "
                    "Use async batch_embeddings instead."
                )
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.batch_embeddings(texts, show_progress, **kwargs)
            )
        finally:
            pass

    async def batch_embeddings(
        self,
        texts: List[str],
        show_progress: bool = True,
        **kwargs,
    ) -> List[List[float]]:
        tasks = [self.embedding(text, **kwargs) for text in texts]

        if show_progress:
            results = await tqdm_asyncio.gather(*tasks, desc="Embeddings")
        else:
            results = await asyncio.gather(*tasks)

        return results

    async def aclose(self) -> None:
        """Close the client and run post-eval hooks."""
        # Run post-eval hooks before closing (pipeline ensures this only runs once)
        if self.adapter_transport is not None:
            try:
                self.adapter_transport.run_post_eval_hooks()
                logger.info("Post-eval hooks executed via aclose()")
            except Exception as e:
                logger.error(f"Failed to run post-eval hooks: {e}")
        await self.client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()
        return False
