"""Agent inference module.

Provides ``make_agent_inference_request`` — the public entrypoint for running
inference against agent endpoints. Routes by ``agent.format`` to format-specific
executors:

- ``generic``: HTTP POST with Jinja-templated body, JSONPath response extraction.
- ``nemo_agent_toolkit``: SSE streaming via ``/generate/full?filter_steps=none``.

Both executors normalise output into an OpenAI-like dict so existing downstream
code (``process_output``, hooks, metrics) works unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable
from typing import Any, Protocol

import httpx
from httpx import Timeout
from jsonpath_ng import parse as jsonpath_parse

from nemo_evaluator.sdk.enums import AgentFormat
from nemo_evaluator.sdk.inference import get_logger, requests_log_var
from nemo_evaluator.sdk.resilience.api import run_with_resilience
from nemo_evaluator.sdk.resilience.classifier import endpoint_identity
from nemo_evaluator.sdk.templates import render_template
from nemo_evaluator.sdk.values.agents import Agent

# Default timeout for agent requests (seconds).
_DEFAULT_TIMEOUT = 120.0


class AgentInferenceFn(Protocol):
    """Callable protocol for agent inference function dependency injection."""

    def __call__(
        self,
        agent: Agent,
        request: dict,
        max_retries: int | None,
        api_key: str | None = None,
        *,
        timeout: float | None = None,
    ) -> Awaitable[dict]: ...


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


async def make_agent_inference_request(
    agent: Agent,
    request: dict,
    max_retries: int | None = 3,
    api_key: str | None = None,
    *,
    timeout: float | None = None,
) -> dict:
    """Run inference against an agent endpoint.

    Routes to the appropriate executor based on ``agent.format``:

    - ``generic`` — HTTP POST with templated body and JSONPath extraction.
    - ``nemo_agent_toolkit`` — SSE streaming via ``/generate/full``.

    Returns a normalised OpenAI-like response dict compatible with
    ``process_output()`` and downstream hooks.
    """
    if agent.format == AgentFormat.NEMO_AGENT_TOOLKIT:
        return await _make_nat_agent_request(agent, request, max_retries=max_retries, api_key=api_key, timeout=timeout)
    else:
        return await _make_generic_agent_request(
            agent, request, max_retries=max_retries, api_key=api_key, timeout=timeout
        )


# ---------------------------------------------------------------------------
# Generic executor
# ---------------------------------------------------------------------------


# TODO: There need to be just one agent inference function, NAT is generic agent with certain fields pre-filled.
async def _make_generic_agent_request(
    agent: Agent,
    request: dict,
    *,
    max_retries: int | None = 3,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Execute inference against a generic agent endpoint.

    1. Render ``agent.body`` Jinja template with the request context.
    2. POST to ``agent.url``.
    3. Extract response text via ``agent.response_path`` JSONPath.
    4. Optionally extract trajectory via ``agent.trajectory_path``.
    """
    log = get_logger()

    resolved_api_key = api_key or agent.api_key
    effective_timeout = timeout or _DEFAULT_TIMEOUT

    # Build context from the incoming request for template rendering.
    context: dict[str, Any] = {**request, "request": request}

    if agent.body is None:
        raise ValueError("body is required for generic agents")
    if agent.response_path is None:
        raise ValueError("response_path is required for generic agents")

    rendered_body = render_template(agent.body, context=context)
    payload = rendered_body if isinstance(rendered_body, dict) else {"args": rendered_body}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    endpoint_key = endpoint_identity(agent.url, model_id=agent.name, auth_identity=resolved_api_key)
    max_attempts = max(1, (max_retries if max_retries is not None else 0) + 1)

    log.info("Making generic agent request to %s", agent.url)

    async with httpx.AsyncClient(timeout=Timeout(effective_timeout)) as client:

        async def _invoke_post() -> dict[str, Any]:
            response = await client.post(agent.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()

        try:
            result_data: dict[str, Any] = await run_with_resilience(
                endpoint_key,
                _invoke_post,
                max_attempts=max_attempts,
            )
        except Exception:
            log.exception("Generic agent request to %s failed after %d attempts", agent.url, max_attempts)
            raise

    # Record request/response for audit
    requests_log = requests_log_var.get([])
    requests_log.append({"request": payload, "response": result_data})

    # Extract response text via JSONPath
    response_text = _extract_jsonpath(result_data, agent.response_path, field_name="response_path")

    # Build normalised response
    normalised: dict[str, Any] = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": str(response_text),
                }
            }
        ]
    }

    # Optionally extract trajectory
    if agent.trajectory_path:
        trajectory = _extract_jsonpath(result_data, agent.trajectory_path, field_name="trajectory_path", required=False)
        if trajectory is not None:
            normalised["trajectory"] = trajectory

    log.info("Generic agent request to %s completed", agent.url)
    return normalised


# ---------------------------------------------------------------------------
# NeMo Agent Toolkit SSE executor
# ---------------------------------------------------------------------------


async def _make_nat_agent_request(
    agent: Agent,
    request: dict,
    *,
    max_retries: int | None = 3,
    api_key: str | None = None,
    timeout: float | None = None,
) -> dict:
    """Execute inference against a NeMo Agent Toolkit endpoint.

    1. Derive ``input_message`` from the request (``messages`` or ``prompt``).
    2. POST to ``{agent.url}/generate/full?filter_steps=none``.
    3. Stream SSE response, capture last ``value`` field.
    4. Return normalised OpenAI-like dict.
    """
    log = get_logger()

    resolved_api_key = api_key or agent.api_key
    effective_timeout = timeout or _DEFAULT_TIMEOUT

    input_message = _derive_input_message(request)

    endpoint = f"{agent.url.rstrip('/')}/generate/full"
    payload = {"input_message": input_message}

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if resolved_api_key:
        headers["Authorization"] = f"Bearer {resolved_api_key}"

    endpoint_key = endpoint_identity(endpoint, model_id=agent.name, auth_identity=resolved_api_key)
    max_attempts = max(1, (max_retries if max_retries is not None else 0) + 1)

    log.info("Making NAT agent request to %s", endpoint)

    async def _invoke_stream() -> str | None:
        result: str | None = None

        async with httpx.AsyncClient(timeout=Timeout(effective_timeout)) as client:
            async with client.stream(
                "POST",
                endpoint,
                json=payload,
                headers=headers,
                params={"filter_steps": "none"},
            ) as response:
                response.raise_for_status()
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])
                            value = chunk_data.get("value")
                            if value is not None:
                                result = value
                        except json.JSONDecodeError:
                            log.debug("Skipping malformed SSE JSON chunk: %s", line)
                            continue

        return result

    try:
        final_response = await run_with_resilience(
            endpoint_key,
            _invoke_stream,
            max_attempts=max_attempts,
        )
    except Exception:
        log.exception("NAT agent request to %s failed after %d attempts", endpoint, max_attempts)
        raise

    if final_response is None:
        raise RuntimeError(
            f"NAT agent at {endpoint} completed the SSE stream without producing a final value. "
            "Verify that the agent endpoint is functioning correctly."
        )

    # Record request/response for audit
    requests_log = requests_log_var.get([])
    requests_log.append({"request": payload, "response": {"value": final_response}})

    log.info("NAT agent request to %s completed", endpoint)

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": final_response,
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _derive_input_message(request: dict) -> str:
    """Derive a single input_message string from an inference request.

    Handles both chat-style (``messages``) and completion-style (``prompt``)
    requests.
    """
    if "messages" in request:
        messages = request["messages"]
        # Use the last user message content, or concatenate all messages
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return str(msg["content"])
        # Fallback: concatenate all message contents
        return "\n".join(str(msg.get("content", "")) for msg in messages)

    if "prompt" in request:
        return str(request["prompt"])

    raise ValueError("Agent inference request must contain 'messages' or 'prompt'.")


def _extract_jsonpath(
    data: dict[str, Any],
    path: str,
    *,
    field_name: str = "path",
    required: bool = True,
) -> Any:
    """Extract a value from data using a JSONPath expression."""
    expr = jsonpath_parse(path)
    matches = expr.find(data)
    if not matches:
        if required:
            raise ValueError(f"JSONPath '{path}' ({field_name}) did not match any value in agent response: {data}")
        return None
    return matches[0].value
