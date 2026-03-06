"""Adapter Server proxy: captures model call trajectories for external environments.

When an external system (Gym, Harbor, PI agent) owns the model call, Evaluator
has no direct visibility into request/response pairs. The proxy solves this by
sitting between the agent and the model endpoint:

    Agent -> Proxy (captures) -> Real Model Endpoint

Usage:
    proxy = AdapterProxy(target_url="https://api.nvidia.com/v1")
    app = proxy.app  # FastAPI app, mount it or run standalone

    # Set the agent's model URL to: http://proxy-host:port/problem/{task_id}/v1/
    # The proxy forwards to the real endpoint and logs every call tagged with task_id.

All captured trajectories are available via proxy.get_trajectories(task_id).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response

from nemo_evaluator.observability.types import ModelResponse, StepRecord

logger = logging.getLogger(__name__)


@dataclass
class ProxyCallRecord:
    task_id: str
    request_body: dict[str, Any]
    response_body: dict[str, Any]
    latency_ms: float
    timestamp: float
    step_record: StepRecord


class AdapterProxy:
    """Reverse proxy that captures model API calls per task_id.

    Forwards all requests to target_url while recording full request/response
    pairs tagged with the task_id from the URL path.

    Path routing: /problem/{task_id}/v1/chat/completions
                  -> {target_url}/v1/chat/completions
    """

    def __init__(self, target_url: str, api_key: str | None = None,
                 timeout: float = 120.0) -> None:
        self.target_url = target_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._calls: dict[str, list[ProxyCallRecord]] = defaultdict(list)
        self._step_idx = 0
        self.app = self._build_app()

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="NeMo Evaluator Adapter Proxy")

        @app.api_route("/problem/{task_id}/{path:path}", methods=["POST", "GET", "PUT"])
        async def proxy_route(task_id: str, path: str, request: Request):
            body_bytes = await request.body()
            try:
                body = await request.json()
            except Exception:
                body = {}

            forward_url = f"{self.target_url}/{path}"
            headers = dict(request.headers)
            headers.pop("host", None)
            if self.api_key:
                headers["authorization"] = f"Bearer {self.api_key}"

            t0 = time.monotonic()
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(
                    method=request.method,
                    url=forward_url,
                    content=body_bytes,
                    headers=headers,
                )
            latency_ms = (time.monotonic() - t0) * 1000

            try:
                resp_body = resp.json()
            except Exception:
                resp_body = {"raw": resp.text[:1000]}

            content = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            model = body.get("model", "")

            choices = resp_body.get("choices", [])
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "")

            usage = resp_body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)

            model_resp = ModelResponse(
                content=content,
                model=model,
                finish_reason=choices[0].get("finish_reason", "") if choices else "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_ms=latency_ms,
                raw_response=resp_body,
            )

            prompt_content = ""
            messages = body.get("messages", [])
            if messages:
                prompt_content = messages[-1].get("content", "")[:500]

            step = StepRecord(
                problem_idx=self._step_idx,
                prompt=prompt_content,
                model_response=model_resp,
                model_ms=latency_ms,
            )
            self._step_idx += 1

            record = ProxyCallRecord(
                task_id=task_id,
                request_body=body,
                response_body=resp_body,
                latency_ms=latency_ms,
                timestamp=time.time(),
                step_record=step,
            )
            self._calls[task_id].append(record)
            logger.debug("Proxy: task=%s latency=%.0fms tokens=%d", task_id, latency_ms, total_tokens)

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
            )

        @app.get("/proxy/trajectories/{task_id}")
        async def get_task_trajectories(task_id: str):
            calls = self._calls.get(task_id, [])
            return {
                "task_id": task_id,
                "n_calls": len(calls),
                "calls": [
                    {
                        "request": c.request_body,
                        "response_content": c.step_record.model_response.content if c.step_record.model_response else "",
                        "latency_ms": c.latency_ms,
                        "tokens": c.step_record.model_response.total_tokens if c.step_record.model_response else 0,
                    }
                    for c in calls
                ],
            }

        @app.get("/proxy/stats")
        async def proxy_stats():
            return {
                "total_calls": sum(len(v) for v in self._calls.values()),
                "tasks_seen": len(self._calls),
                "task_call_counts": {k: len(v) for k, v in self._calls.items()},
            }

        return app

    def get_trajectories(self, task_id: str) -> list[StepRecord]:
        return [c.step_record for c in self._calls.get(task_id, [])]

    def get_all_steps(self) -> list[StepRecord]:
        steps = []
        for calls in self._calls.values():
            steps.extend(c.step_record for c in calls)
        return sorted(steps, key=lambda s: s.timestamp)

    def clear(self) -> None:
        self._calls.clear()
        self._step_idx = 0
