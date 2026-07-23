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
"""HarborSolver: runs Harbor-compatible agents inside an evaluator Sandbox."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import copy
import json
import logging
import os
import random
import shlex
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.errors import GracefulError, InfraError
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import ErrorKind, SolveResult
from nemo_evaluator.solvers.compaction_logging import (
    CompactionAttempt,
    CompactionEvent,
    CompactionMechanism,
    CompactionStrategy,
    CompactionTokens,
    CompactionTokensIntermediate,
    CompactionTrigger,
)
from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import SeedResult
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_INFRA_ERROR_NAMES = frozenset(
    {
        "ServiceUnavailableError",
        "ConnectionError",
        "TimeoutError",
        "ConnectError",
        "ReadTimeout",
        "APIConnectionError",
        # TODO: decide whether litellm request timeouts ("Timeout" / "APITimeoutError")
        # belong here. Classifying them as infra makes them retryable, but it also
        # raises InfraError ahead of the `agent_timed_out and workspace_diff` branch
        # that submits a partial patch for verification — so a timeout that already
        # produced a usable workspace diff would be discarded instead of scored.
        # Deferred to a follow-up to keep this change from altering scoring.
    }
)


def _resolve_agent_timeout(
    strategy: str,
    config_timeout: float,
    task_timeout: float | None,
    max_cap: float | None,
) -> float:
    """Compute effective agent timeout based on strategy.

    Args:
        strategy: "override" (config wins), "task" (per-task from task.toml),
                  or "max" (larger of both).
        config_timeout: timeout from NEL config (solver.run_timeout or bench.timeout).
        task_timeout: per-task timeout from task.toml ``[agent] timeout_sec``.
        max_cap: optional hard ceiling (``max_agent_timeout`` config field).
    """
    if strategy == "override" or task_timeout is None:
        result = config_timeout
    elif strategy == "task":
        result = task_timeout
    elif strategy == "max":
        result = max(config_timeout, task_timeout)
    else:
        logger.warning("Unknown timeout_strategy '%s', falling back to config_timeout", strategy)
        result = config_timeout

    if max_cap is not None:
        result = min(result, max_cap)
    return result


def _extract_response(context: Any) -> str:
    """Extract text response from AgentContext (metadata > last rollout)."""
    if context.metadata and isinstance(context.metadata.get("response"), str):
        return context.metadata["response"]
    if context.rollout_details:
        last = context.rollout_details[-1]
        c = last.get("content") if isinstance(last, dict) else getattr(last, "content", None)
        if isinstance(c, str):
            return c
    return ""


# ---------------------------------------------------------------------------
# API key resolution
# ---------------------------------------------------------------------------


def _resolve_api_key(explicit: str | None) -> str | None:
    """Return *explicit* if non-empty, else probe env vars in order:
    ``LLM_API_KEY``, ``NVIDIA_API_KEY``, ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``.
    """
    if explicit:
        return explicit
    for var in ("LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _model_id_for_openai(model_id: str, has_custom_url: bool, *, agent: str = "") -> str:
    """Return *model_id* with an ``openai/`` prefix when needed.

    The prefix tells LiteLLM to use the OpenAI-compatible provider when
    routing through a custom endpoint URL.  Agents that don't go through
    LiteLLM (e.g. ``claude-code``, which talks to the Anthropic API
    directly) must not receive the prefix.
    """
    if agent.lower() == "claude-code":
        return model_id
    if has_custom_url and not model_id.startswith("openai/"):
        return f"openai/{model_id}"
    return model_id


def _ensure_host_env(api_key: str, model_id: str | None, *, has_custom_url: bool) -> None:
    """Populate host-process env vars required by OpenHands-family agents.

    ``OpenHandsSDK.run()`` reads ``LLM_API_KEY`` (required) and
    ``LLM_MODEL`` (fallback) directly from ``os.environ`` before
    building the container exec environment.  Uses ``setdefault`` so
    values set by an earlier solver or caller are never overwritten.

    ``LLM_BASE_URL`` is intentionally *not* set here; each ``solve()``
    call passes a per-session URL via the adapter's ``override_env``.
    """
    os.environ.setdefault("LLM_API_KEY", api_key)
    if model_id:
        os.environ.setdefault("LLM_MODEL", _model_id_for_openai(model_id, has_custom_url))
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ.setdefault("LITELLM_TELEMETRY", "false")


def _ensure_claude_host_env(api_key: str, base_url: str) -> None:
    """Populate host-process env vars read by the ``claude-code`` Harbor agent.

    The agent shells out to ``claude`` which reads ``ANTHROPIC_API_KEY``
    (required) and ``ANTHROPIC_BASE_URL`` (optional — used to route
    through NVIDIA's inference API).  ``setdefault`` preserves anything
    the user already exported.
    """
    if api_key:
        os.environ.setdefault("ANTHROPIC_API_KEY", api_key)
    if base_url:
        os.environ.setdefault("ANTHROPIC_BASE_URL", base_url)


# ---------------------------------------------------------------------------
# Agent log download
# ---------------------------------------------------------------------------


async def _download_agent_logs(
    sandbox: "Sandbox",
    dest: Path,
    *,
    max_retries: int = 3,
    timeout: float = 180.0,
) -> None:
    """Download ``/logs/agent/`` from the container into *dest*.

    Retries on transient failures and enforces a wall-clock timeout so a
    hung exec server cannot block the solver indefinitely.
    """
    try:
        async with asyncio.timeout(timeout):
            await _download_agent_logs_inner(sandbox, dest, max_retries=max_retries)
    except TimeoutError:
        logger.warning(
            "Agent log download timed out after %.0fs — trajectory may be incomplete",
            timeout,
        )
    except Exception:
        logger.error("Agent log download failed", exc_info=True)


async def _download_agent_logs_inner(
    sandbox: "Sandbox",
    dest: Path,
    *,
    max_retries: int = 3,
) -> None:
    ls = await sandbox.exec(
        "find /logs/agent -type f 2>/dev/null | head -80",
        timeout_sec=15,
    )
    if not ls.stdout:
        logger.warning("No files in /logs/agent/ inside container")
        return
    logger.info("Container /logs/agent/:\n%s", ls.stdout.strip())

    remote_tar = "/tmp/_eval_agent_logs.tar.gz"
    rc = await sandbox.exec(
        f"tar czf {remote_tar} -C /logs/agent .",
        timeout_sec=120,
    )
    if rc.return_code != 0:
        logger.error("tar failed (rc=%d): %s", rc.return_code, rc.stderr or rc.stdout)
        return

    import tarfile

    fd, _tmp = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    local_tar = Path(_tmp)
    try:
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                await sandbox.download(remote_tar, local_tar)
                dest.mkdir(parents=True, exist_ok=True)
                with tarfile.open(local_tar, "r:gz") as tar:
                    tar.extractall(dest, filter="data")
                logger.info("Downloaded %d files to %s", len(list(dest.rglob("*"))), dest)
                return
            except Exception as exc:
                last_err = exc
                if attempt < max_retries:
                    logger.warning(
                        "Agent log download attempt %d/%d failed: %s",
                        attempt,
                        max_retries,
                        exc,
                    )
                    await asyncio.sleep(2.0 * attempt)
                else:
                    logger.error(
                        "Agent log download failed after %d attempts: %s",
                        max_retries,
                        last_err,
                    )
    finally:
        local_tar.unlink(missing_ok=True)
        try:
            await sandbox.exec(f"rm -f {remote_tar}", timeout_sec=10)
        except Exception:
            pass


def _openhands_compaction_module_files() -> dict[str, bytes]:
    solvers_dir = Path(__file__).parent
    return {
        "/installed-agent/nel/nemo_evaluator/__init__.py": b"",
        "/installed-agent/nel/nemo_evaluator/solvers/__init__.py": b"",
        "/installed-agent/nel/nemo_evaluator/solvers/compaction_logging.py": (
            solvers_dir / "compaction_logging.py"
        ).read_bytes(),
        "/installed-agent/nel/nemo_evaluator/solvers/openhands_compaction.py": (
            solvers_dir / "openhands_compaction.py"
        ).read_bytes(),
    }


async def _patch_openhands_sdk(sandbox: Sandbox, *, cmd_timeout: float | None = None) -> None:
    """Apply runtime patches to the OpenHands SDK inside the sandbox.

    1. **Prevent premature FINISHED on text-only responses** — when the
       LLM returns text without a tool call the SDK sets
       ``execution_status = FINISHED`` and stops.  Reasoning models often
       produce intermediate text before their next tool call, so the
       patched loop continues until ``finish`` is called or
       ``max_iterations`` is reached.

    2. **Capture reasoning, metrics, and tool timing in ATIF trajectory** —
       the runner's event serialization drops reasoning, per-turn token
       usage, and observation timestamps.  The event conversion and
       ``build_trajectory`` stages are patched to preserve them.

    3. **Enforce 300 s hard timeout on terminal commands** — the SDK
       only applies a hard timeout when the model passes an explicit
       ``timeout`` parameter.  The patch imposes a 300 s ceiling on
       every command to prevent a single long-running process from
       consuming the entire ``run_timeout`` budget.

    4. **Disable default visualizer + stuck detection** — the SDK's
       ``DefaultConversationVisualizer`` renders every event through
       ``rich``, whose grapheme splitter is pathologically slow on long
       text containing U+200B adjacent to URLs.  Observed on 26 django
       SWE-bench prompts: the first ``send_message()`` burns 100 % CPU
       inside ``rich.cells.split_graphemes`` for the full 90 min
       ``run_timeout`` without ever reaching the LLM.  ``stuck_detection``
       is also disabled in the same patch because its heuristic
       mis-flags reasoning-model turns.

    5. **Honor ``LLM_TIMEOUT`` in the Harbor runner** — older Harbor
       images ship ``run_agent.py`` without reading ``LLM_TIMEOUT``.
       Inject the env lookup so ``solver.agent_kwargs.llm_kwargs.timeout``
       (mapped to ``container_env.LLM_TIMEOUT``) reaches OpenHands
       ``LLM(timeout=...)`` and LiteLLM per-request timeouts.

    6. **Log OpenHands condensation in ATIF trajectories** — upload NEL
       compaction helpers and splice system compaction steps after
       ``build_trajectory()`` when ``Condensation`` events are present.

    7. **Exclude condensation-only steps from ``max_iteration_per_run``** —
       when ``agent.step`` returns after emitting ``Condensation``, do not
       increment the conversation iteration counter so compaction does not
       consume the agent step budget.
    """
    # -- Patch 0: disable stuck detection + default visualizer -----------
    # stuck_detection=False: the SDK's heuristic mis-flags reasoning-model
    # loops that produce text-only turns followed by the next tool call.
    #
    # visualizer=None: the SDK's DefaultConversationVisualizer renders each
    # event through ``rich`` on send_message/run, and rich's grapheme
    # splitter (``rich.cells.split_graphemes`` / ``chop_cells``) is
    # pathologically slow on long text containing zero-width characters
    # (U+200B) adjacent to URLs. A handful of SWE-bench prompts (notably
    # 26 django tasks carrying ``\u200b`` + GitHub deep-links) would
    # deadlock the first ``conversation.send_message()`` call at 100%
    # CPU inside rich — no syscalls, no LLM request ever sent, the whole
    # agent stalls until the outer run_timeout (90 min) fires. Visual
    # output is irrelevant in eval runs, so disable the visualizer
    # entirely.
    _runner_patch_script = (
        "p = '/installed-agent/run_agent.py'\n"
        "c = open(p).read()\n"
        'old = \'    conv_kwargs: dict[str, Any] = {"agent": agent, "workspace": workspace}\\n\'\n'
        'new = old + \'    conv_kwargs["stuck_detection"] = False\\n    conv_kwargs["visualizer"] = None\\n\'\n'
        "already = 'conv_kwargs[\"visualizer\"] = None' in c\n"
        "if old in c and not already:\n"
        "    c = c.replace(old, new, 1)\n"
        "    open(p, 'w').write(c)\n"
        "    already = True\n"
        "if already:\n"
        "    print('stuck_detection disabled, visualizer disabled')\n"
        "else:\n"
        "    print('pattern not found')\n"
    )
    encoded0 = base64.b64encode(_runner_patch_script.encode()).decode()
    r0 = await sandbox.exec(
        f"echo {encoded0} | base64 -d | python3",
        timeout_sec=10,
    )
    logger.info("Runner patch: %s", (r0.stdout or "").strip())

    _llm_timeout_patch_script = """\
p = '/installed-agent/run_agent.py'
c = open(p).read()
if 'LLM_TIMEOUT' in c:
    print('llm_timeout: already present')
else:
    old = '    llm = LLM(**llm_kwargs)'
    new = (
        '    import os as _llm_timeout_os\\n'
        '    timeout_raw = _llm_timeout_os.environ.get("LLM_TIMEOUT")\\n'
        '    if timeout_raw:\\n'
        '        llm_kwargs["timeout"] = int(timeout_raw)\\n'
        '    llm = LLM(**llm_kwargs)'
    )
    if old in c:
        open(p, 'w').write(c.replace(old, new, 1))
        print('llm_timeout: patched')
    else:
        print('llm_timeout: pattern not found')
"""
    encoded_timeout = base64.b64encode(_llm_timeout_patch_script.encode()).decode()
    r_timeout = await sandbox.exec(
        f"echo {encoded_timeout} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout_timeout = (r_timeout.stdout or "").strip()
    logger.info("LLM timeout patch: %s", stdout_timeout)
    if r_timeout.return_code != 0 or "pattern not found" in stdout_timeout:
        logger.warning(
            "LLM timeout patch problem (rc=%d): %s",
            r_timeout.return_code,
            stdout_timeout or (r_timeout.stderr or "")[:300],
        )

    # -- Patch 1: don't FINISHED on text-only responses ------------------
    # In agent.py, when the LLM produces text without a tool call the SDK
    # sets execution_status=FINISHED and returns — killing the agent.
    # We replace the 6-line block with a no-op so execution continues
    # until finish() or max_iterations.  We also widen the nudge to
    # always fire (not just for empty responses), so the model knows it
    # must use a tool.
    _agent_patch_script = """\
import glob
fs = glob.glob('/opt/openhands-sdk-venv/lib/python*/site-packages/openhands/sdk/agent/agent.py')
p = fs[0] if fs else ''
assert p, 'agent.py not found'
c = open(p).read()

old1 = (
    '        # Finish conversation if LLM produced content (awaits user input)\\n'
    '        # Continue if only reasoning without content (e.g., GPT-5 codex thinking)\\n'
    '        if has_content:\\n'
    '            logger.debug("LLM produced a message response - awaits user input")\\n'
    '            state.execution_status = ConversationExecutionStatus.FINISHED\\n'
    '            return'
)
new1 = (
    '        # [NEL] text-only response: continue instead of FINISHED\\n'
    '        if has_content:\\n'
    '            logger.debug("LLM produced text without tool call - continuing (NEL)")'
)
old2 = '        if not has_content:'
new2 = '        if True:  # [NEL] always nudge when no tool call'

ok1 = old1 in c
c2 = c.replace(old1, new1, 1) if ok1 else c
ok2 = old2 in c2
c3 = c2.replace(old2, new2, 1) if ok2 else c2
if ok1 or ok2:
    open(p, 'w').write(c3)
print(f'agent.py FINISHED={ok1} nudge={ok2} at {p}')
"""
    encoded1 = base64.b64encode(_agent_patch_script.encode()).decode()
    r2 = await sandbox.exec(
        f"echo {encoded1} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout2 = (r2.stdout or "").strip()
    logger.info("Agent.py patch: %s", stdout2)
    if r2.return_code != 0 or "False" in stdout2:
        logger.warning(
            "Agent.py patch problem (rc=%d): %s",
            r2.return_code,
            stdout2 or (r2.stderr or "")[:300],
        )

    _max_iter_patch_script = """\
import glob
fs = glob.glob('/opt/openhands-sdk-venv/lib/python*/site-packages/openhands/sdk/conversation/impl/local_conversation.py')
p = fs[0] if fs else ''
assert p, 'local_conversation.py not found'
c = open(p).read()
if '_nel_last = self._state.events[-1]' in c:
    print(f'local_conversation.py max_iter condensation skip: already patched at {p}')
else:
    old_imp = (
        'from openhands.sdk.event import (\\n'
        '    ActionEvent,\\n'
        '    AgentErrorEvent,\\n'
        '    CondensationRequest,\\n'
    )
    new_imp = (
        'from openhands.sdk.event import (\\n'
        '    ActionEvent,\\n'
        '    AgentErrorEvent,\\n'
        '    Condensation,\\n'
        '    CondensationRequest,\\n'
    )
    ok_imp = old_imp in c
    if ok_imp:
        c = c.replace(old_imp, new_imp, 1)
    old_inc = (
        '                    finally:\\n'
        '                        self._step_holds_state_lock = False\\n'
        '                    iteration += 1\\n'
    )
    new_inc = (
        '                    finally:\\n'
        '                        self._step_holds_state_lock = False\\n'
        '                    _nel_last = self._state.events[-1] if self._state.events else None\\n'
        '                    if not isinstance(_nel_last, Condensation):\\n'
        '                        iteration += 1\\n'
    )
    n_inc = c.count(old_inc)
    if n_inc:
        c = c.replace(old_inc, new_inc)
    old_inc_async = (
        '                        finally:\\n'
        '                            self._step_holds_state_lock = False\\n'
        '                        iteration += 1\\n'
    )
    new_inc_async = (
        '                        finally:\\n'
        '                            self._step_holds_state_lock = False\\n'
        '                        _nel_last = self._state.events[-1] if self._state.events else None\\n'
        '                        if not isinstance(_nel_last, Condensation):\\n'
        '                            iteration += 1\\n'
    )
    n_inc_async = c.count(old_inc_async)
    if n_inc_async:
        c = c.replace(old_inc_async, new_inc_async)
    if ok_imp or n_inc or n_inc_async:
        open(p, 'w').write(c)
    print(f'local_conversation.py max_iter condensation skip: import={ok_imp} sync={n_inc} async={n_inc_async} at {p}')
"""
    encoded_max_iter = base64.b64encode(_max_iter_patch_script.encode()).decode()
    r_max_iter = await sandbox.exec(
        f"echo {encoded_max_iter} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout_max_iter = (r_max_iter.stdout or "").strip()
    logger.info("Max-iteration condensation skip patch: %s", stdout_max_iter)
    if r_max_iter.return_code != 0 or (
        "already patched" not in stdout_max_iter and "sync=0" in stdout_max_iter and "async=0" in stdout_max_iter
    ):
        logger.warning(
            "Max-iteration condensation skip patch problem (rc=%d): %s",
            r_max_iter.return_code,
            stdout_max_iter or (r_max_iter.stderr or "")[:300],
        )

    # -- Patch 2: capture reasoning, metrics, and tool timing in ATIF --------
    # The runner converts SDK events → intermediate dicts → ATIF steps but
    # never copies reasoning, per-turn token usage, or observation timestamps.
    #
    # Event-collection (A, B): extract reasoning_content and LLM usage from
    # the SDK event object and store in the intermediate dict.
    # build_trajectory (C, D): propagate step fields and observation timing.

    _reasoning_script = """\
import sys
p = '/installed-agent/run_agent.py'
c = open(p).read()

# A: MessageEvent agent - extract reasoning + usage
old_a = (
    '                events_list.append(entry)\\n'
    '                last_agent_timestamp = event.timestamp\\n'
    '        elif isinstance(event, ActionEvent):'
)
new_a = (
    '                _rc = getattr(event, "reasoning_content", "") or ""\\n'
    '                if not _rc and hasattr(event, "llm_message"):\\n'
    '                    _rc = getattr(event.llm_message, "reasoning_content", "") or ""\\n'
    '                if _rc:\\n'
    '                    entry["reasoning_content"] = _rc\\n'
    '                _lm_resp_id = getattr(event, "llm_response_id", None)\\n'
    '                if _lm_resp_id:\\n'
    '                    try:\\n'
    '                        _nel_seen_resp_ids\\n'
    '                    except NameError:\\n'
    '                        _nel_seen_resp_ids = set()\\n'
    '                    if _lm_resp_id not in _nel_seen_resp_ids:\\n'
    '                        _tu = next((u for u in getattr(getattr(llm, "metrics", None), "token_usages", []) if getattr(u, "response_id", None) == _lm_resp_id), None)\\n'
    '                        if _tu:\\n'
    '                            entry["usage"] = {"prompt_tokens": int(getattr(_tu, "prompt_tokens", 0)), "completion_tokens": int(getattr(_tu, "completion_tokens", 0))}\\n'
    '                            _nel_seen_resp_ids.add(_lm_resp_id)\\n'
    '                events_list.append(entry)\\n'
    '                last_agent_timestamp = event.timestamp\\n'
    '        elif isinstance(event, ActionEvent):'
)

# B: ActionEvent - extract reasoning + usage
old_b = (
    '            events_list.append(entry)\\n'
    '            last_agent_timestamp = event.timestamp\\n'
    '        elif isinstance(event, ObservationEvent):'
)
new_b = (
    '            _rc = getattr(event, "reasoning_content", "") or ""\\n'
    '            if not _rc:\\n'
    '                _tp = getattr(event, "thought", [])\\n'
    '                if _tp:\\n'
    '                    _rc = chr(10).join(getattr(c, "text", str(c)) for c in _tp if getattr(c, "text", None))\\n'
    '            if _rc:\\n'
    '                entry["reasoning_content"] = _rc\\n'
    '            _lm_resp_id = getattr(event, "llm_response_id", None)\\n'
    '            if _lm_resp_id:\\n'
    '                try:\\n'
    '                    _nel_seen_resp_ids\\n'
    '                except NameError:\\n'
    '                    _nel_seen_resp_ids = set()\\n'
    '                if _lm_resp_id not in _nel_seen_resp_ids:\\n'
    '                    _tu = next((u for u in getattr(getattr(llm, "metrics", None), "token_usages", []) if getattr(u, "response_id", None) == _lm_resp_id), None)\\n'
    '                    if _tu:\\n'
    '                        entry["usage"] = {"prompt_tokens": int(getattr(_tu, "prompt_tokens", 0)), "completion_tokens": int(getattr(_tu, "completion_tokens", 0))}\\n'
    '                        _nel_seen_resp_ids.add(_lm_resp_id)\\n'
    '            events_list.append(entry)\\n'
    '            last_agent_timestamp = event.timestamp\\n'
    '        elif isinstance(event, ObservationEvent):'
)

# C: build_trajectory - propagate reasoning + metrics to ATIF steps
old_c = (
    '            steps.append(step)\\n'
    '            step_id += 1\\n'
    '\\n'
    '        elif event_type == "tool_result":'
)
new_c = (
    '            _rc = event.get("reasoning_content", "")\\n'
    '            if _rc:\\n'
    '                step["reasoning_content"] = _rc\\n'
    '            _u = event.get("usage")\\n'
    '            if isinstance(_u, dict):\\n'
    '                step["metrics"] = {"prompt_tokens": int(_u.get("prompt_tokens", 0) or 0), "completion_tokens": int(_u.get("completion_tokens", 0) or 0)}\\n'
    '            steps.append(step)\\n'
    '            step_id += 1\\n'
    '\\n'
    '        elif event_type == "tool_result":'
)

# D: build_trajectory - preserve observation timing
old_d = (
    '        elif event_type == "tool_result":\\n'
    '            # Find the previous step and add observation\\n'
    '            if steps and steps[-1].get("source") == "agent":\\n'
    '                steps[-1]["observation"] = {\\n'
    '                    "results": [\\n'
    '                        {\\n'
    '                            "source_call_id": event.get("tool_call_id"),\\n'
    '                            "content": event.get("content", ""),\\n'
    '                        }\\n'
    '                    ]\\n'
    '                }'
)
new_d = (
    '        elif event_type == "tool_result":\\n'
    '            if steps and steps[-1].get("source") == "agent":\\n'
    '                _started_at = steps[-1].get("timestamp")\\n'
    '                _completed_at = event.get("timestamp")\\n'
    '                _timing = {\\n'
    '                    key: value\\n'
    '                    for key, value in (("started_at", _started_at), ("completed_at", _completed_at))\\n'
    '                    if value\\n'
    '                }\\n'
    '                if _started_at and _completed_at:\\n'
    '                    try:\\n'
    '                        from datetime import datetime as _datetime\\n'
    '                        _start = _datetime.fromisoformat(str(_started_at).replace("Z", "+00:00"))\\n'
    '                        _end = _datetime.fromisoformat(str(_completed_at).replace("Z", "+00:00"))\\n'
    '                        _timing["duration_ms"] = max(0.0, (_end - _start).total_seconds() * 1000)\\n'
    '                    except (TypeError, ValueError):\\n'
    '                        pass\\n'
    '                _result = {\\n'
    '                    "source_call_id": event.get("tool_call_id"),\\n'
    '                    "content": event.get("content", ""),\\n'
    '                }\\n'
    '                if _timing:\\n'
    '                    _result["extra"] = _timing\\n'
    '                steps[-1]["observation"] = {"results": [_result]}'
)

old_schema = '"schema_version": "ATIF-v1.5",'
new_schema = '"schema_version": "ATIF-v1.7",'

# Enable the LLM summarizing condenser when OH_CONDENSER_MAX_SIZE is set.
old_cond_build = '    agent = Agent(**agent_kwargs)'
new_cond_build = (
    '    _cms = os.environ.get("OH_CONDENSER_MAX_SIZE")\\n'
    '    if _cms:\\n'
    '        from openhands.sdk.context.condenser import LLMSummarizingCondenser\\n'
    '        _hdrs = dict(llm.extra_headers or {})\\n'
    '        _hdrs["x-nel-call-kind"] = "compaction"\\n'
    '        _cond_llm = llm.model_copy(update={"extra_headers": _hdrs, "usage_id": f"{llm.usage_id}-compaction"})\\n'
    '        _ck = {"llm": _cond_llm, "max_size": int(_cms), "keep_first": 4}\\n'
    '        _cmt = os.environ.get("OH_CONDENSER_MAX_TOKENS")\\n'
    '        if _cmt:\\n'
    '            _ck["max_tokens"] = int(_cmt)\\n'
    '        agent_kwargs["condenser"] = LLMSummarizingCondenser(**_ck)\\n'
    '        print(f"Condenser enabled: max_size={_cms} max_tokens={_cmt}")\\n'
    '    agent = Agent(**agent_kwargs)'
)

# Import the Condensation event type (used by the condensation logging below).
old_cond_import = (
    '    ActionEvent,\\n'
    '    MessageEvent,'
)
new_cond_import = (
    '    ActionEvent,\\n'
    '    Condensation,\\n'
    '    MessageEvent,'
)

# Initialise condensation tracking + per-event word counts before the event loop.
old_cond_init = (
    '    last_agent_timestamp: str | None = None\\n'
    '    for event in conversation.state.events:\\n'
    '        if isinstance(event, MessageEvent):'
)
new_cond_init = (
    '    last_agent_timestamp: str | None = None\\n'
    '    event_words_by_id = {}\\n'
    '    _cond_records = []\\n'
    '    for event in conversation.state.events:\\n'
    '        event_words_by_id[getattr(event, "id", "")] = len(str(event).split())\\n'
    '        if isinstance(event, MessageEvent):'
)

# Record each condensation event and persist the totals to metrics.
old_cond_branch = (
    '                        break\\n'
    '\\n'
    '    # Build and save trajectory\\n'
    '    trajectory = build_trajectory('
)
new_cond_branch = (
    '                        break\\n'
    '        elif isinstance(event, Condensation):\\n'
    '            _forgotten = list(event.forgotten_event_ids or [])\\n'
    '            _summary = event.summary or ""\\n'
    '            _cond_records.append({"forgotten_events": len(_forgotten), "forgotten_words": sum(event_words_by_id.get(fid, 0) for fid in _forgotten), "summary_words": len(_summary.split()), "summary_chars": len(_summary), "success": bool(_summary.strip())})\\n'
    '            print(f"Condensation #{len(_cond_records)}: forgot {len(_forgotten)} events -> summary {len(_summary.split())} words ({len(_summary)} chars), success={bool(_summary.strip())}")\\n'
    '\\n'
    '    if _cond_records:\\n'
    '        metrics["condensations"] = len(_cond_records)\\n'
    '        metrics["condensation_details"] = _cond_records\\n'
    '\\n'
    '    # Build and save trajectory\\n'
    '    trajectory = build_trajectory('
)

# Surface the condensation summary in build_trajectory's final_metrics.
old_cond_final = (
    '    }\\n'
    '\\n'
    '    return trajectory'
)
new_cond_final = (
    '    }\\n'
    '\\n'
    '    if llm_metrics.get("condensations"):\\n'
    '        trajectory["final_metrics"]["condensations"] = llm_metrics["condensations"]\\n'
    '        trajectory["final_metrics"]["condensation_details"] = llm_metrics.get("condensation_details", [])\\n'
    '\\n'
    '    return trajectory'
)

ok_a = old_a in c
c = c.replace(old_a, new_a, 1) if ok_a else c
ok_b = old_b in c
c = c.replace(old_b, new_b, 1) if ok_b else c
ok_c = old_c in c
c = c.replace(old_c, new_c, 1) if ok_c else c
ok_d = old_d in c
c = c.replace(old_d, new_d, 1) if ok_d else c
ok_schema = old_schema in c
c = c.replace(old_schema, new_schema, 1) if ok_d and ok_schema else c
ok_cond_build = old_cond_build in c
c = c.replace(old_cond_build, new_cond_build, 1) if ok_cond_build else c
ok_cond_import = old_cond_import in c
c = c.replace(old_cond_import, new_cond_import, 1) if ok_cond_import else c
ok_cond_init = old_cond_init in c
c = c.replace(old_cond_init, new_cond_init, 1) if ok_cond_init else c
ok_cond_branch = old_cond_branch in c
c = c.replace(old_cond_branch, new_cond_branch, 1) if ok_cond_branch else c
ok_cond_final = old_cond_final in c
c = c.replace(old_cond_final, new_cond_final, 1) if ok_cond_final else c
open(p, 'w').write(c)
print(f'reasoning+metrics+timing: msg={ok_a} action={ok_b} traj={ok_c} timing={ok_d} schema={ok_schema}')
print(f'condenser: build={ok_cond_build} import={ok_cond_import} init={ok_cond_init} branch={ok_cond_branch} final={ok_cond_final}')
"""
    encoded = base64.b64encode(_reasoning_script.encode()).decode()
    r3 = await sandbox.exec(
        f"echo {encoded} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout3 = (r3.stdout or "").strip()
    logger.info("Reasoning+metrics patch: %s", stdout3)
    if r3.return_code != 0 or "False" in stdout3:
        logger.warning(
            "Reasoning+metrics patch problem (rc=%d): %s",
            r3.return_code,
            stdout3 or (r3.stderr or "")[:300],
        )

    # -- Patch 3: hard timeout ceiling on terminal commands ------------------
    if cmd_timeout is not None and cmd_timeout > 0:
        _cmd_timeout_script = f"""\
import glob, sys
fs = glob.glob(
    '/opt/openhands-sdk-venv/lib/python*/site-packages/'
    'openhands/tools/terminal/terminal/terminal_session.py'
)
p = fs[0] if fs else ''
assert p, 'terminal_session.py not found'
c = open(p).read()

_MAX = {cmd_timeout!r}

old = (
    '            if action.timeout is not None:\\n'
    '                time_since_start = time.time() - start_time\\n'
    '                if time_since_start >= action.timeout:\\n'
    '                    obs = self._handle_hard_timeout_command(\\n'
    '                        command,\\n'
    '                        terminal_content=cur_terminal_output,\\n'
    '                        ps1_matches=ps1_matches,\\n'
    '                        timeout=action.timeout,\\n'
    '                    )\\n'
    '                    logger.debug(f\\"RETURNING OBSERVATION (hard-timeout): {{obs}}\\")\\n'
    '                    return obs'
)

new = (
    '            _NEL_MAX = ' + str(_MAX) + '  # [NEL] hard ceiling on any command\\n'
    '            _eff_timeout = (\\n'
    '                min(action.timeout, _NEL_MAX)\\n'
    '                if action.timeout is not None\\n'
    '                else _NEL_MAX\\n'
    '            )\\n'
    '            if elapsed_time >= _eff_timeout:\\n'
    '                obs = self._handle_hard_timeout_command(\\n'
    '                    command,\\n'
    '                    terminal_content=cur_terminal_output,\\n'
    '                    ps1_matches=ps1_matches,\\n'
    '                    timeout=_eff_timeout,\\n'
    '                )\\n'
    '                logger.debug(f\\"RETURNING OBSERVATION (hard-timeout): {{obs}}\\")\\n'
    '                return obs'
)

ok = old in c
if ok:
    c = c.replace(old, new, 1)
    open(p, 'w').write(c)
print(f'cmd_timeout_{{_MAX}}s={{ok}} at {{p}}')
"""
        encoded5 = base64.b64encode(_cmd_timeout_script.encode()).decode()
        r5 = await sandbox.exec(
            f"echo {encoded5} | base64 -d | python3",
            timeout_sec=10,
        )
        stdout5 = (r5.stdout or "").strip()
        logger.info("Cmd timeout patch (%ss): %s", cmd_timeout, stdout5)
        if r5.return_code != 0 or "False" in stdout5:
            logger.warning(
                "Cmd timeout patch problem (rc=%d): %s",
                r5.return_code,
                stdout5 or (r5.stderr or "")[:300],
            )
    else:
        logger.info("Cmd timeout patch: skipped (cmd_timeout not configured)")

    # -- Patch 4: flush trajectory when the runner is interrupted ----------
    # Anchor 1: wrap `conversation.send_message()` + `conversation.run()` in
    # try/except BaseException so main() continues to the existing
    # event-reconstruction + build_trajectory + save code on timeout/crash.
    # SIGTERM/SIGINT are converted to a catchable BaseException so evaluator
    # timeout cancellation gets the same flush path.  On interruption, first
    # write a cheap partial trajectory from already-received model events.  The
    # normal full writer below overwrites it if post-run serialization completes.
    # Anchor 2: after the final print in main(), exit(0) cleanly so Harbor
    # treats the run as a normal completion and downloads the trajectory.
    _budget_flush_script = """\
import sys
p = '/installed-agent/run_agent.py'
c = open(p).read()

old1 = (
    '    # Send instruction and run\\n'
    '    conversation.send_message(args.instruction)\\n'
    '    conversation.run()'
)
ind = '    '

old2 = None
for line in c.splitlines():
    if 'Total cost:' in line and 'print' in line:
        old2 = line; break

already = '_trajectory_flush_exc' in c
ok = old1 in c and old2 is not None and not already
success = already or ok
print(f'anchor1={repr(old1)} anchor2={repr(old2)} already={already} ok={ok}')

if ok:
    partial = (
        ind + 'def _trajectory_text_from_event(_event):\\n' +
        ind + '    _msg = getattr(_event, "llm_message", None)\\n' +
        ind + '    _raw = getattr(_msg, "content", None) if _msg is not None else None\\n' +
        ind + '    if isinstance(_raw, list):\\n' +
        ind + '        return chr(10).join(getattr(_c, "text", str(_c)) for _c in _raw if getattr(_c, "text", None))\\n' +
        ind + '    return str(_raw) if _raw else ""\\n' +
        ind + 'def _trajectory_tool_args(_event):\\n' +
        ind + '    if getattr(_event, "tool_call", None) and hasattr(_event.tool_call, "function"):\\n' +
        ind + '        _raw_args = getattr(_event.tool_call.function, "arguments", None)\\n' +
        ind + '        if isinstance(_raw_args, str):\\n' +
        ind + '            try:\\n' +
        ind + '                return json.loads(_raw_args)\\n' +
        ind + '            except json.JSONDecodeError:\\n' +
        ind + '                return {"raw": _raw_args}\\n' +
        ind + '        if isinstance(_raw_args, dict):\\n' +
        ind + '            return _raw_args\\n' +
        ind + '    if getattr(_event, "action", None):\\n' +
        ind + '        try:\\n' +
        ind + '            _ad = _event.action.model_dump() if hasattr(_event.action, "model_dump") else vars(_event.action)\\n' +
        ind + '            return {_k: _v for _k, _v in _ad.items() if _k != "kind" and _v is not None}\\n' +
        ind + '        except Exception:\\n' +
        ind + '            pass\\n' +
        ind + '    return {}\\n' +
        ind + 'def _write_partial_trajectory(_reason):\\n' +
        ind + '    import json, os, sys\\n' +
        ind + '    from pathlib import Path\\n' +
        ind + '    try:\\n' +
        ind + '        _metric_obj = getattr(llm, "metrics", None)\\n' +
        ind + '        _usage = getattr(_metric_obj, "accumulated_token_usage", None)\\n' +
        ind + '        _metrics = {\\n' +
        ind + '            "prompt_tokens": int(getattr(_usage, "prompt_tokens", 0) or 0),\\n' +
        ind + '            "completion_tokens": int(getattr(_usage, "completion_tokens", 0) or 0),\\n' +
        ind + '            "cached_tokens": int(getattr(_usage, "cache_read_tokens", 0) or 0),\\n' +
        ind + '            "cost_usd": float(getattr(_metric_obj, "accumulated_cost", 0.0) or 0.0),\\n' +
        ind + '        }\\n' +
        ind + '        _events_list = []\\n' +
        ind + '        for _event in list(getattr(getattr(conversation, "state", None), "events", []) or []):\\n' +
        ind + '            if isinstance(_event, MessageEvent):\\n' +
        ind + '                if _event.source in ("user", "agent"):\\n' +
        ind + '                    _entry_type = "assistant_message" if _event.source == "agent" else "user_message"\\n' +
        ind + '                    _entry = {"type": _entry_type, "content": _trajectory_text_from_event(_event), "timestamp": _event.timestamp}\\n' +
        ind + '                    _events_list.append(_entry)\\n' +
        ind + '            elif isinstance(_event, ActionEvent):\\n' +
        ind + '                _events_list.append({"type": "assistant_message", "content": "", "timestamp": _event.timestamp, "tool_calls": [{"id": _event.tool_call_id, "name": _event.tool_name, "arguments": _trajectory_tool_args(_event)}]})\\n' +
        ind + '        _trajectory = build_trajectory(_events_list, _metrics, model)\\n' +
        ind + '        _trajectory.setdefault("extra", {})["partial_trajectory"] = {"reason": _reason, "events": len(_events_list)}\\n' +
        ind + '        _path = Path(args.trajectory_path)\\n' +
        ind + '        _path.parent.mkdir(parents=True, exist_ok=True)\\n' +
        ind + '        _tmp = _path.with_suffix(_path.suffix + ".partial")\\n' +
        ind + '        with open(_tmp, "w") as _f:\\n' +
        ind + '            json.dump(_trajectory, _f, indent=2)\\n' +
        ind + '        os.replace(_tmp, _path)\\n' +
        ind + '        print(f"trajectory flush: partial trajectory saved to {_path}", file=sys.stderr, flush=True)\\n' +
        ind + '        return True\\n' +
        ind + '    except Exception as _save_e:\\n' +
        ind + '        print(f"trajectory flush: partial trajectory save failed: {type(_save_e).__name__}: {_save_e}", file=sys.stderr, flush=True)\\n' +
        ind + '        return False\\n'
    )
    wrap = (
        ind + '# Send instruction and run\\n' +
        ind + '_trajectory_flush_exc = None\\n' +
        partial +
        ind + 'class TrajectoryFlushRequested(BaseException):\\n' +
        ind + '    pass\\n' +
        ind + 'def _trajectory_timeout_handler(_signum, _frame):\\n' +
        ind + '    raise TrajectoryFlushRequested(f"signal {_signum}")\\n' +
        ind + 'try:\\n' +
        ind + '    import signal as _trajectory_signal\\n' +
        ind + '    _trajectory_signal.signal(_trajectory_signal.SIGTERM, _trajectory_timeout_handler)\\n' +
        ind + '    _trajectory_signal.signal(_trajectory_signal.SIGINT, _trajectory_timeout_handler)\\n' +
        ind + 'except Exception:\\n' +
        ind + '    pass\\n' +
        ind + 'try:\\n' +
        ind + '    conversation.send_message(args.instruction)\\n' +
        ind + '    conversation.run()\\n' +
        ind + 'except TrajectoryFlushRequested as _e:\\n' +
        ind + '    _trajectory_flush_exc = _e\\n' +
        ind + '    print(f"trajectory flush: flushing after NEL timeout signal: {_e}", file=sys.stderr, flush=True)\\n' +
        ind + '    _write_partial_trajectory("timeout")\\n' +
        ind + 'except BaseException as _e:\\n' +
        ind + '    _trajectory_flush_exc = _e\\n' +
        ind + '    print(f"trajectory flush: flushing after {type(_e).__name__}: {_e}", file=sys.stderr, flush=True)\\n' +
        ind + '    _write_partial_trajectory(type(_e).__name__)'
    )
    c = c.replace(old1, wrap, 1)
    exit_block = (
        ind + 'if _trajectory_flush_exc is not None:\\n' +
        ind + '    if not isinstance(_trajectory_flush_exc, TrajectoryFlushRequested):\\n' +
        ind + '        import json as _j, os as _os\\n' +
        ind + '        _os.makedirs("/logs/agent", exist_ok=True)\\n' +
        ind + '        _marker = "/logs/agent/agent_error.json"\\n' +
        ind + '        _marker_tmp = _marker + ".tmp"\\n' +
        ind + '        with open(_marker_tmp, "w") as _mf:\\n' +
        ind + '            _mf.write(_j.dumps({"etype": type(_trajectory_flush_exc).__name__, "emsg": str(_trajectory_flush_exc)}))\\n' +
        ind + '        _os.replace(_marker_tmp, _marker)\\n' +
        ind + '    sys.exit(0)')
    c = c.replace(old2, old2 + '\\n' + exit_block, 1)
    open(p, 'w').write(c)
print(f'budget_flush={success}')
"""
    encoded4 = base64.b64encode(_budget_flush_script.encode()).decode()
    r4 = await sandbox.exec(
        f"echo {encoded4} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout4 = (r4.stdout or "").strip()
    logger.info("Budget-flush patch: %s", stdout4)
    if r4.return_code != 0 or "budget_flush=True" not in stdout4:
        logger.warning(
            "Budget-flush patch problem (rc=%d): %s",
            r4.return_code,
            stdout4 or (r4.stderr or "")[:300],
        )

    _compaction_files_payload = {
        path: base64.b64encode(content).decode() for path, content in _openhands_compaction_module_files().items()
    }
    _compaction_upload_script = (
        "import base64, json, os\n"
        f"files = json.loads({json.dumps(_compaction_files_payload)!r})\n"
        "for path, content_b64 in files.items():\n"
        "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
        "    with open(path, 'wb') as f:\n"
        "        f.write(base64.b64decode(content_b64))\n"
        "print('nel_compaction_modules=ok')\n"
    )
    encoded_compaction_upload = base64.b64encode(_compaction_upload_script.encode()).decode()
    r_compaction_upload = await sandbox.exec(
        f"echo {encoded_compaction_upload} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout_compaction_upload = (r_compaction_upload.stdout or "").strip()
    logger.info("OpenHands compaction module upload: %s", stdout_compaction_upload)
    if r_compaction_upload.return_code != 0 or "ok" not in stdout_compaction_upload:
        logger.warning(
            "OpenHands compaction module upload problem (rc=%d): %s",
            r_compaction_upload.return_code,
            stdout_compaction_upload or (r_compaction_upload.stderr or "")[:300],
        )

    _compaction_inject_script = """\
import sys
p = '/installed-agent/run_agent.py'
c = open(p).read()

old = (
    '    trajectory = build_trajectory(\\n'
    '        events_list,\\n'
    '        metrics,\\n'
    '        model,\\n'
    '        system_prompt=system_prompt,\\n'
    '        tool_definitions=tool_definitions,\\n'
    '    )\\n'
    '\\n'
    '    trajectory_path = Path(args.trajectory_path)'
)

new = (
    '    trajectory = build_trajectory(\\n'
    '        events_list,\\n'
    '        metrics,\\n'
    '        model,\\n'
    '        system_prompt=system_prompt,\\n'
    '        tool_definitions=tool_definitions,\\n'
    '    )\\n'
    '    if "/installed-agent/nel" not in sys.path:\\n'
    '        sys.path.insert(0, "/installed-agent/nel")\\n'
    '    from nemo_evaluator.solvers.openhands_compaction import enrich_trajectory_with_compaction\\n'
    '    _ctx_lim_raw = os.environ.get("CONTEXT_LIMIT") or os.environ.get("MAX_INPUT_TOKENS") or "128000"\\n'
    '    _ctx_lim = int(_ctx_lim_raw)\\n'
    '    _oh_cfg = {}\\n'
    '    if os.environ.get("OH_CONDENSER_MAX_SIZE"):\\n'
    '        _oh_cfg["max_size"] = int(os.environ["OH_CONDENSER_MAX_SIZE"])\\n'
    '    if os.environ.get("OH_CONDENSER_MAX_TOKENS"):\\n'
    '        _oh_cfg["max_tokens"] = int(os.environ["OH_CONDENSER_MAX_TOKENS"])\\n'
    '    if os.environ.get("OH_CONDENSER_KEEP_FIRST"):\\n'
    '        _oh_cfg["keep_first"] = int(os.environ["OH_CONDENSER_KEEP_FIRST"])\\n'
    '    trajectory = enrich_trajectory_with_compaction(\\n'
    '        trajectory, conversation.state.events, llm, _ctx_lim, condenser_config=_oh_cfg or None,\\n'
    '    )\\n'
    '\\n'
    '    trajectory_path = Path(args.trajectory_path)'
)

already = 'enrich_trajectory_with_compaction' in c
ok = old in c and not already
if ok:
    c = c.replace(old, new, 1)
    open(p, 'w').write(c)
print(f'openhands_compaction_inject={ok} already={already}')
"""
    encoded_compaction_inject = base64.b64encode(_compaction_inject_script.encode()).decode()
    r_compaction_inject = await sandbox.exec(
        f"echo {encoded_compaction_inject} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout_compaction_inject = (r_compaction_inject.stdout or "").strip()
    logger.info("OpenHands compaction inject: %s", stdout_compaction_inject)
    if r_compaction_inject.return_code != 0 or "openhands_compaction_inject=False" in stdout_compaction_inject:
        logger.warning(
            "OpenHands compaction inject problem (rc=%d): %s",
            r_compaction_inject.return_code,
            stdout_compaction_inject or (r_compaction_inject.stderr or "")[:300],
        )


# ---------------------------------------------------------------------------
# Trajectory / token / response recovery  (agent-agnostic)
# ---------------------------------------------------------------------------
#
# Each Harbor agent writes its own logs and converts them to ATIF via
# ``populate_context_post_run()``.  The evaluator:
#   1. Reads the ATIF trajectory.json the agent produced.
#   2. Falls back to the largest .txt as an error log if nothing structured exists.
#
# Agent-specific parsing (OpenHands completions/, sessions/events/, etc.)
# is the agent's responsibility.
# ---------------------------------------------------------------------------


def _flatten_atif_continuation_chain(root: dict[str, Any], agent_logs_dir: Path) -> dict[str, Any]:
    merged = copy.deepcopy(root)
    steps: list[dict[str, Any]] = list(merged.get("steps") or [])
    current = root
    while True:
        continued_ref = current.get("continued_trajectory_ref")
        if not continued_ref or not isinstance(continued_ref, str):
            break
        next_path = agent_logs_dir / continued_ref
        if not next_path.is_file():
            logger.warning(
                "continued_trajectory_ref %r points to missing file %s",
                continued_ref,
                next_path,
            )
            break
        try:
            continuation = json.loads(next_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read continuation trajectory %s: %s", next_path, exc)
            break
        if not isinstance(continuation, dict):
            break
        cont_steps = continuation.get("steps")
        if isinstance(cont_steps, list):
            steps.extend(copy.deepcopy(cont_steps))
        current = continuation

    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            step["step_id"] = index

    merged["steps"] = steps
    merged.pop("continued_trajectory_ref", None)
    final_metrics = current.get("final_metrics")
    if isinstance(final_metrics, dict):
        merged["final_metrics"] = copy.deepcopy(final_metrics)
    return merged


def _recover_from_logs(agent_logs_dir: Path) -> dict[str, Any]:
    """Read trajectory + token counts from *agent_logs_dir*.

    Returns ``{"trajectory": [...], "prompt_tokens": int,
    "completion_tokens": int, "response": str}``.
    """
    out: dict[str, Any] = {
        "trajectory": [],
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "response": "",
    }

    # -- 1. ATIF trajectory JSON (canonical agent output) -------------------
    traj_files = sorted(
        (f for f in agent_logs_dir.glob("*.json") if "traj" in f.stem.lower()),
        key=lambda p: (p.name != "trajectory.json", p.name),
    )
    canonical = agent_logs_dir / "trajectory.json"
    if canonical.is_file() and canonical not in traj_files:
        traj_files.insert(0, canonical)

    for tf in traj_files:
        try:
            raw = json.loads(tf.read_text())
        except Exception:
            logger.warning("Unreadable trajectory file: %s", tf)
            continue

        parsed = _parse_atif(raw)
        if parsed:
            doc = parsed["doc"]
            if tf.name == "trajectory.json":
                doc = _flatten_atif_continuation_chain(doc, agent_logs_dir)
                merged_parse = _parse_atif(doc)
                if merged_parse:
                    parsed = merged_parse
            logger.info("Trajectory loaded from %s", tf.name)
            out["trajectory"] = [doc]
            out["prompt_tokens"] = parsed["prompt_tokens"]
            out["completion_tokens"] = parsed["completion_tokens"]
            out["response"] = parsed["response"]
            return out

        logger.warning(
            "%s is not ATIF — skipping (agent should convert in populate_context_post_run)",
            tf.name,
        )

    # -- 2. Largest .txt log as generic fallback ----------------------------
    txt_files = sorted(
        agent_logs_dir.rglob("*.txt"),
        key=lambda p: p.stat().st_size,
        reverse=True,
    )
    for txt in txt_files:
        try:
            text = txt.read_text(errors="replace")
        except Exception:
            continue
        if text.strip():
            rel = txt.relative_to(agent_logs_dir)
            logger.info("Using %s as error log trajectory", rel)
            lines = text.strip().splitlines()
            if len(lines) > 200:
                summary = "\n".join(lines[:50]) + "\n\n... [truncated middle] ...\n\n" + "\n".join(lines[-100:])
            else:
                summary = text.strip()
            out["trajectory"] = build_atif_trajectory(
                steps=[
                    {
                        "source": "system",
                        "message": summary,
                        "extra": {"source_file": str(rel)},
                    }
                ],
                status="error",
            )
            return out

    return out


def _error_from_crash_marker(agent_logs_dir: Path) -> str | None:
    """Return a user-facing crash error from the agent sidecar, or raise infra errors."""
    crash_file = agent_logs_dir / "agent_error.json"
    if not crash_file.is_file():
        return None

    try:
        crash = json.loads(crash_file.read_text())
        etype = crash.get("etype", "AgentCrash")
        emsg = crash.get("emsg", "")
        if etype in _INFRA_ERROR_NAMES:
            raise InfraError(f"Agent infrastructure failure: {etype}: {emsg}")
        return f"Agent crashed: {etype}: {emsg}"
    except InfraError:
        raise
    except Exception:
        logger.warning("HarborSolver: failed to read crash marker", exc_info=True)
        return None


def _parse_atif(raw: Any) -> dict[str, Any] | None:
    """Return parsed ATIF document or ``None`` if *raw* is not ATIF."""
    if not isinstance(raw, dict):
        return None
    if not (str(raw.get("schema_version", "")).startswith("ATIF") or ("steps" in raw and "agent" in raw)):
        return None

    fm = raw.get("final_metrics") or {}
    response = ""
    for step in reversed(raw.get("steps", [])):
        msg = step.get("message") or step.get("content") or ""
        if isinstance(msg, str) and msg.strip():
            response = msg
            break

    return {
        "doc": raw,
        "prompt_tokens": fm.get("total_prompt_tokens", 0),
        "completion_tokens": fm.get("total_completion_tokens", 0),
        "response": response,
    }


# ---------------------------------------------------------------------------
# Workspace diff capture + prompt-echo detection
# ---------------------------------------------------------------------------


async def _capture_workspace_diff(sandbox: "Sandbox") -> str:
    """Run ``git diff HEAD`` inside the agent container and return the diff.

    Returns empty string on any failure.  This gives the solver a meaningful
    response for benchmarks where the agent modifies source code (SWE-bench,
    terminal-bench, etc.) rather than producing text.
    """
    for workdir in ("/testbed", "/app", "/workspace"):
        result = await sandbox.exec(
            f"cd {workdir} && git diff HEAD 2>/dev/null",
            timeout_sec=30,
        )
        if result.return_code == 0 and result.stdout.strip():
            diff = result.stdout.strip()
            if len(diff) > 50_000:
                diff = diff[:50_000] + "\n... [diff truncated at 50 KB]"
            return diff
    return ""


def _is_prompt_echo(response: str, prompt: str) -> bool:
    """True when *response* is just the task prompt echoed back.

    Agents that work by modifying files (rather than producing text)
    sometimes set their "response" to the original instruction.
    """
    if not response or not prompt:
        return False
    r = response.strip()
    p = prompt.strip()
    if r == p:
        return True
    if len(r) > 200 and len(p) > 200:
        return r[:200] == p[:200] and r[-200:] == p[-200:]
    return False


# ---------------------------------------------------------------------------
# HarborSolver
# ---------------------------------------------------------------------------


def _check_harbor_installed() -> None:
    try:
        import harbor  # noqa: F401
    except ImportError:
        raise ImportError(
            "Harbor agent integration requires the harbor package. Install with: pip install nemo-evaluator[harbor]"
        ) from None
    _silence_harbor_debug()


def _silence_harbor_debug(level: int = logging.INFO) -> None:
    """Override harbor's per-module DEBUG levels.

    Harbor's ``setup_logger`` hardcodes ``setLevel(DEBUG)`` on every logger
    it creates, which overrides any parent-level setting.  We walk the
    logger registry and reset all ``harbor.*`` loggers.
    """
    logging.getLogger("harbor").setLevel(level)
    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("harbor"):
            logging.getLogger(name).setLevel(level)


def _patch_terminus_tmux_send_keys() -> None:
    """Make Harbor's Terminus 2 ``tmux send-keys`` chunking byte-accurate.

    Harbor's ``TmuxSession._tmux_send_keys`` splits keystrokes so each emitted
    ``tmux send-keys`` command stays under ``_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH``,
    but it measures length with ``len()`` (Unicode code points) while tmux's
    limit is in **bytes**.  Multibyte UTF-8 payloads (e.g. Lean proofs full of
    ``ℕ ∀ ∃``) therefore produce commands that pass the character-count check yet
    exceed tmux's byte limit, raising ``command too long`` — the agent crashes,
    verify is skipped, and the trial scores 0.
    """
    try:
        from harbor.agents.terminus_2.tmux_session import TmuxSession
    except ImportError:
        logger.debug("Harbor terminus_2 TmuxSession not importable; skipping tmux byte-length patch")
        return

    if getattr(TmuxSession, "_nel_tmux_bytelen_patched", False):
        return

    missing = [
        attr
        for attr in ("_tmux_send_keys", "_split_key_for_tmux", "_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH")
        if not hasattr(TmuxSession, attr)
    ]
    if missing:
        logger.warning(
            "Harbor TmuxSession is missing %s; skipping tmux send-keys byte-length patch "
            "(Harbor internals may have changed — review nemo_evaluator.solvers.harbor).",
            missing,
        )
        return

    def _utf8_len(value: str) -> int:
        return len(value.encode("utf-8"))

    def _tmux_send_keys(self: Any, keys: list[str]) -> list[str]:
        prefix = "tmux send-keys -t " + shlex.quote(self._session_name)
        prefix_len = _utf8_len(prefix)
        max_len = self._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH

        escaped_keys = [shlex.quote(key) for key in keys]
        single = prefix + " " + " ".join(escaped_keys)
        if _utf8_len(single) <= max_len:
            return [single]

        commands: list[str] = []
        current_escaped: list[str] = []
        current_len = prefix_len

        def _flush() -> None:
            nonlocal current_len
            if current_escaped:
                commands.append(prefix + " " + " ".join(current_escaped))
                current_escaped.clear()
                current_len = prefix_len

        for key in keys:
            escaped = shlex.quote(key)
            addition = 1 + _utf8_len(escaped)  # space + quoted key

            if current_len + addition <= max_len:
                current_escaped.append(escaped)
                current_len += addition
            elif prefix_len + addition <= max_len:
                _flush()
                current_escaped.append(escaped)
                current_len = prefix_len + addition
            else:
                _flush()
                max_escaped = max_len - prefix_len - 1
                for chunk_escaped in _split_key_for_tmux(key, max_escaped):
                    addition = 1 + _utf8_len(chunk_escaped)
                    if current_len + addition <= max_len:
                        current_escaped.append(chunk_escaped)
                        current_len += addition
                    else:
                        _flush()
                        current_escaped.append(chunk_escaped)
                        current_len = prefix_len + addition

        _flush()
        return commands

    def _split_key_for_tmux(key: str, max_escaped_len: int) -> list[str]:
        """Split *key* into ``shlex.quote``-d chunks each ≤ *max_escaped_len* bytes."""
        chunks: list[str] = []
        remaining = key
        while remaining:
            lo, hi, best = 1, len(remaining), 0
            while lo <= hi:
                mid = (lo + hi) // 2
                if _utf8_len(shlex.quote(remaining[:mid])) <= max_escaped_len:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            if best == 0:
                raise ValueError("tmux send-keys command prefix leaves no room for key")
            chunks.append(shlex.quote(remaining[:best]))
            remaining = remaining[best:]
        return chunks

    TmuxSession._utf8_len = staticmethod(_utf8_len)
    TmuxSession._tmux_send_keys = _tmux_send_keys
    TmuxSession._split_key_for_tmux = staticmethod(_split_key_for_tmux)
    TmuxSession._nel_tmux_bytelen_patched = True
    logger.info("Applied Harbor terminus-2 tmux send-keys UTF-8 byte-length patch")


_TERMINUS_CLE_RESET_PATCHED = False
_TERMINUS_QUERY_LLM_SRC: str | None = None

_TERMINUS_CLE_RESET_REPLACEMENTS = [
    (
        "            summary_prompt = None\n            # Fallback 1: Try full summary\n",
        "            summary_prompt = None\n"
        "            full_summarize_failed_with_cle = False\n"
        "            # Fallback 1: Try full summary\n",
    ),
    (
        "            except Exception as e:\n"
        '                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")\n',
        "            except Exception as e:\n"
        "                full_summarize_failed_with_cle = isinstance(\n"
        "                    e, ContextLengthExceededError\n"
        "                )\n"
        '                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")\n',
    ),
    (
        "            if prompt_path is not None:\n"
        "                prompt_path.write_text(summary_prompt)\n"
        "\n"
        "            try:\n"
        "                start_time = time.time()\n"
        "                llm_response = await chat.chat(\n",
        "            if prompt_path is not None:\n"
        "                prompt_path.write_text(summary_prompt)\n"
        "\n"
        "            if full_summarize_failed_with_cle:\n"
        "                chat._messages = [chat.messages[0]]\n"
        "                chat.reset_response_chain()\n"
        "\n"
        "            try:\n"
        "                start_time = time.time()\n"
        "                llm_response = await chat.chat(\n",
    ),
    (
        "                llm_response = LLMResponse(\n"
        '                    content="Technical difficulties. Please continue with the task."\n'
        "                )\n",
        "                llm_response = LLMResponse(\n"
        "                    content=self._build_local_fallback_llm_content()\n"
        "                )\n",
    ),
]


def _terminus2_build_local_fallback_llm_content(self) -> str:
    import json

    analysis = "Technical difficulties during context recovery. All summarization fallbacks failed."
    plan = "Technical difficulties. Please continue with the task."
    if self._parser_name == "xml":
        return (
            "<response>\n"
            f"<analysis>{analysis}</analysis>\n"
            f"<plan>{plan}</plan>\n"
            "<commands>\n"
            "</commands>\n"
            "<task_complete>false</task_complete>\n"
            "</response>"
        )
    return json.dumps(
        {
            "analysis": analysis,
            "plan": plan,
            "commands": [],
            "task_complete": False,
        }
    )


def _patch_terminus_cle_reset() -> None:
    global _TERMINUS_CLE_RESET_PATCHED, _TERMINUS_QUERY_LLM_SRC
    if _TERMINUS_CLE_RESET_PATCHED:
        return

    import inspect
    import textwrap

    from harbor.agents.terminus_2 import terminus_2 as terminus2_mod

    original = terminus2_mod.Terminus2._query_llm
    if _TERMINUS_QUERY_LLM_SRC is None:
        _TERMINUS_QUERY_LLM_SRC = inspect.getsource(inspect.unwrap(original))
    src = _TERMINUS_QUERY_LLM_SRC

    if "full_summarize_failed_with_cle" in src:
        _TERMINUS_CLE_RESET_PATCHED = True
        logger.info("Terminus2._query_llm already contains the CLE chat-reset logic; skipping patch")
        return

    for anchor, replacement in _TERMINUS_CLE_RESET_REPLACEMENTS:
        occurrences = src.count(anchor)
        if occurrences != 1:
            raise RuntimeError(
                "Cannot patch Terminus2._query_llm for CLE chat reset: expected exactly "
                f"one match for an anchor but found {occurrences}. Harbor's terminus_2.py "
                "has diverged from the expected source."
            )
        src = src.replace(anchor, replacement, 1)

    if not hasattr(terminus2_mod.Terminus2, "_build_local_fallback_llm_content"):
        terminus2_mod.Terminus2._build_local_fallback_llm_content = _terminus2_build_local_fallback_llm_content

    namespace: dict[str, Any] = {}
    exec(textwrap.dedent(src), terminus2_mod.__dict__, namespace)
    terminus2_mod.Terminus2._query_llm = namespace["_query_llm"]
    _TERMINUS_QUERY_LLM_SRC = src
    _TERMINUS_CLE_RESET_PATCHED = True
    logger.info("Patched Terminus2._query_llm: reset chat on full-summary CLE + parseable local fallback")


_TERMINUS_UNWIND_MIN_PAIRS = 10
_TERMINUS_UNWIND_PATCHED = False

_TERMINUS_UNWIND_REPLACEMENTS = [
    (
        "        context_limit = self._llm.get_model_context_limit()\n"
        "\n"
        "        while len(chat.messages) > 1:  # Keep at least the first message\n",
        "        context_limit = self._llm.get_model_context_limit()\n"
        "\n"
        f"        min_pairs_to_remove = {_TERMINUS_UNWIND_MIN_PAIRS}\n"
        "        pairs_removed = 0\n"
        "        while len(chat.messages) > 1:  # Keep at least the first message\n",
    ),
    (
        "            if free_tokens >= target_free_tokens:\n                break\n",
        "            if free_tokens >= target_free_tokens and pairs_removed >= min_pairs_to_remove:\n"
        "                break\n",
    ),
    (
        "            if len(chat.messages) >= 2:\n"
        "                chat._messages = chat.messages[:-2]\n"
        "            else:\n"
        "                break\n",
        "            if len(chat.messages) >= 2:\n"
        "                chat._messages = chat.messages[:-2]\n"
        "                pairs_removed += 1\n"
        "            else:\n"
        "                break\n",
    ),
    (
        "        chat.reset_response_chain()\n"
        "        free_tokens = context_limit - self._count_total_tokens(chat)\n"
        "        self.logger.debug(\n",
        "        chat.reset_response_chain()\n"
        "        self._nel_last_unwind_pairs = pairs_removed\n"
        "        self._nel_last_unwind_target = target_free_tokens\n"
        "        free_tokens = context_limit - self._count_total_tokens(chat)\n"
        "        self.logger.debug(\n",
    ),
]


def _patch_terminus_unwind_min_pairs() -> None:
    global _TERMINUS_UNWIND_PATCHED
    if _TERMINUS_UNWIND_PATCHED:
        return

    import inspect
    import textwrap

    from harbor.agents.terminus_2 import terminus_2 as terminus2_mod

    original = terminus2_mod.Terminus2._unwind_messages_to_free_tokens
    src = inspect.getsource(inspect.unwrap(original))

    if "min_pairs_to_remove" in src:
        _TERMINUS_UNWIND_PATCHED = True
        logger.info("Terminus2._unwind_messages_to_free_tokens already enforces a minimum pair removal; skipping patch")
        return

    for anchor, replacement in _TERMINUS_UNWIND_REPLACEMENTS:
        occurrences = src.count(anchor)
        if occurrences != 1:
            raise RuntimeError(
                "Cannot patch Terminus2._unwind_messages_to_free_tokens for minimum pair "
                f"removal: expected exactly one match for an anchor but found {occurrences}. "
                "Harbor's terminus_2.py has diverged from the expected source."
            )
        src = src.replace(anchor, replacement, 1)

    namespace: dict[str, Any] = {}
    exec(textwrap.dedent(src), terminus2_mod.__dict__, namespace)
    terminus2_mod.Terminus2._unwind_messages_to_free_tokens = namespace["_unwind_messages_to_free_tokens"]
    _TERMINUS_UNWIND_PATCHED = True
    logger.info(
        "Patched Terminus2._unwind_messages_to_free_tokens: always remove at least %d message pairs",
        _TERMINUS_UNWIND_MIN_PAIRS,
    )


_TERMINUS_API_ANCHOR_PATCHED = False
_CHAT_TOKEN_ANCHOR_PATCHED = False
_HARBOR_LITELLM_CTXLIM_PATCHED = False


def _terminus2_count_total_tokens(self, chat) -> int:
    from litellm.utils import token_counter

    messages = chat.messages
    anchor = getattr(chat, "_api_token_anchor", None)
    if anchor is None:
        if not getattr(chat, "_api_anchor_warned", False):
            chat._api_anchor_warned = True
            logger.warning(
                "No API token-usage anchor available for terminus-2; falling back to litellm's "
                "tokenizer, which may provide inaccurate token count estimates."
            )
        return token_counter(model=self._model_name, messages=messages)

    anchor_len, anchor_total = anchor
    if len(messages) > anchor_len:
        return anchor_total + token_counter(model=self._model_name, messages=messages[anchor_len:])
    if len(messages) == anchor_len:
        return anchor_total
    if not getattr(chat, "_api_anchor_below_warned", False):
        chat._api_anchor_below_warned = True
        logger.debug(
            "terminus-2 messages (%d) shrank below the API anchor (%d); using litellm's tokenizer for token count estimation",
            len(messages),
            anchor_len,
        )
    return token_counter(model=self._model_name, messages=messages)


def _patch_chat_token_anchor() -> None:
    global _CHAT_TOKEN_ANCHOR_PATCHED
    if _CHAT_TOKEN_ANCHOR_PATCHED:
        return

    from harbor.llms.chat import Chat

    if getattr(Chat, "_records_api_token_anchor", False):
        _CHAT_TOKEN_ANCHOR_PATCHED = True
        return

    original_chat = Chat.chat

    async def _chat_with_token_anchor(self, *args, **kwargs):
        response = await original_chat(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
            self._api_token_anchor = (len(self._messages), usage.prompt_tokens + usage.completion_tokens)
        return response

    Chat.chat = _chat_with_token_anchor
    Chat._records_api_token_anchor = True
    _CHAT_TOKEN_ANCHOR_PATCHED = True
    logger.info("Patched Chat.chat to record the API token anchor (prompt + completion) after each call")


def _patch_terminus_api_token_anchor() -> None:
    global _TERMINUS_API_ANCHOR_PATCHED
    if _TERMINUS_API_ANCHOR_PATCHED:
        return

    import inspect

    from harbor.agents.terminus_2 import terminus_2 as terminus2_mod

    src = inspect.getsource(inspect.unwrap(terminus2_mod.Terminus2._count_total_tokens))
    if "token_counter(model=self._model_name, messages=chat.messages)" not in src:
        raise RuntimeError(
            "Cannot patch Terminus2._count_total_tokens for API-anchored token counting: "
            "Harbor's terminus_2.py has diverged from the expected source."
        )

    _patch_chat_token_anchor()
    terminus2_mod.Terminus2._count_total_tokens = _terminus2_count_total_tokens
    _TERMINUS_API_ANCHOR_PATCHED = True
    logger.info("Patched Terminus2._count_total_tokens: anchor on API usage + litellm token remainder")


# Matches the vLLM ``VLLMValidationError`` phrasing raised by self-hosted vLLM
# deployments when a request exceeds the configured context length. The raw 400
# body reads:
#
#   "You passed N input tokens and requested 0 output tokens. However, the
#    model's context length is only M tokens..."
#
# Harbor 0.3.x's ``LiteLLM._is_context_length_error`` phrases tuple lists
# ``"maximum context length"`` but NOT ``"model's context length"``, so this
# text falls through to a plain ``LiteLLMBadRequestError``, Terminus-2's
# ``except ContextLengthExceededError`` block never fires, and the agent
# crashes instead of running its reactive-summarization fallback.
#
# Harbor v0.18.0 added exactly this substring upstream (see
# ``harbor/llms/lite_llm.py`` with the comment ``# vllm 0.16.0 error
# message``), so this patch becomes redundant the moment the ``harbor`` pin
# in ``pyproject.toml`` is bumped to a release containing it.
_VLLM_CTXLIM_PHRASES: tuple[str, ...] = (
    "model's context length",  # vLLM VLLMValidationError; harbor >=0.18 catches this natively
)

# vLLM's ``VLLMValidationError`` is raised with ``parameter="input_tokens"`` as
# a structured kwarg. On stringification, that appears verbatim in the wire
# error body as ``parameter=input_tokens``. Grep of vLLM shows this kwarg is
# emitted by a single call site — the context-overflow validator — so its
# presence in an unclassified ``BadRequestError`` is a strong signal that the
# free-text phrase list above has drifted behind a vLLM release. We do NOT
# reclassify on this marker (a bad free-text detector should be fixed, not
# silently papered over); instead we log loudly so the drift is visible.
_VLLM_CTXLIM_DRIFT_MARKER = "parameter=input_tokens"


def _patch_harbor_lite_llm_context_length_matcher() -> None:
    global _HARBOR_LITELLM_CTXLIM_PATCHED
    if _HARBOR_LITELLM_CTXLIM_PATCHED:
        return

    from harbor.llms import lite_llm as harbor_litellm

    original = harbor_litellm.LiteLLM._is_context_length_error

    def _patched(self: Any, error: Any) -> bool:
        if original(self, error):
            return True
        parts = [
            str(error),
            str(getattr(error, "body", "") or ""),
            str(getattr(error, "message", "") or ""),
            str(getattr(error, "error", "") or ""),
        ]
        combined = " ".join(p for p in parts if p).lower()
        if any(phrase in combined for phrase in _VLLM_CTXLIM_PHRASES):
            return True
        if _VLLM_CTXLIM_DRIFT_MARKER in combined:
            logger.warning(
                "BadRequestError body contains vLLM's stable ctx-overflow marker "
                "'%s' but did not match any classifier phrase in "
                "_VLLM_CTXLIM_PHRASES. vLLM's error phrasing has likely drifted "
                "and _VLLM_CTXLIM_PHRASES needs an update. Body preview: %s",
                _VLLM_CTXLIM_DRIFT_MARKER,
                combined[:500],
            )
        return False

    harbor_litellm.LiteLLM._is_context_length_error = _patched
    _HARBOR_LITELLM_CTXLIM_PATCHED = True
    logger.info('Patched Harbor LiteLLM._is_context_length_error to recognize vLLM "model\'s context length" phrasing')


_TERMINUS_COMPACTION_PATCHED = False


def _terminus2_nel_ensure_compaction_state(self) -> None:
    if not hasattr(self, "_nel_pending_compaction"):
        self._nel_pending_compaction = None
    if not hasattr(self, "_nel_last_unwind_pairs"):
        self._nel_last_unwind_pairs = 0
    if not hasattr(self, "_nel_last_unwind_target"):
        self._nel_last_unwind_target = None
    if not hasattr(self, "_nel_cle_chat_reset_applied"):
        self._nel_cle_chat_reset_applied = False
    if not hasattr(self, "_nel_stashed_tokens_before"):
        self._nel_stashed_tokens_before = None
    if not hasattr(self, "_nel_stashed_tokens_after"):
        self._nel_stashed_tokens_after = None
    if not hasattr(self, "_nel_stashed_tokens_intermediate"):
        self._nel_stashed_tokens_intermediate = None
    if not hasattr(self, "_nel_compaction_count"):
        self._nel_compaction_count = 0


def _terminus2_nel_stash_compaction_token_snapshots(
    self,
    tokens_before: CompactionTokens,
    tokens_after: CompactionTokens,
    tokens_intermediate: CompactionTokensIntermediate | None,
) -> None:
    self._nel_ensure_compaction_state()
    self._nel_stashed_tokens_before = tokens_before
    self._nel_stashed_tokens_after = tokens_after
    self._nel_stashed_tokens_intermediate = tokens_intermediate


def _terminus2_nel_token_snapshot(self, chat) -> CompactionTokens:
    from nemo_evaluator.solvers.compaction_logging import CompactionTokens

    context_limit = self._llm.get_model_context_limit()
    current_tokens = self._count_total_tokens(chat)
    return CompactionTokens(
        prompt_tokens_approx=current_tokens,
        context_limit=context_limit,
        free_tokens=context_limit - current_tokens,
    )


def _terminus2_nel_compaction_mechanism(self) -> CompactionMechanism:
    from nemo_evaluator.solvers.compaction_logging import CompactionMechanism

    unwind_pairs = int(getattr(self, "_nel_last_unwind_pairs", 0) or 0)
    return CompactionMechanism(
        unwind_applied=unwind_pairs > 0,
        chat_reset_applied=bool(getattr(self, "_nel_cle_chat_reset_applied", False)),
        messages_unwound_pairs=unwind_pairs,
        unwind_target_free_tokens=getattr(self, "_nel_last_unwind_target", None),
    )


def _terminus2_nel_make_compaction_event(
    self,
    *,
    trigger: CompactionTrigger,
    strategy: CompactionStrategy,
    replacement_content: str,
    subagent_refs: Any,
    attempts: list[CompactionAttempt] | None,
    tokens_before: CompactionTokens,
    tokens_after: CompactionTokens,
    tokens_intermediate: CompactionTokensIntermediate | None = None,
    handoff_prompt: str | None = None,
) -> CompactionEvent:
    from nemo_evaluator.solvers.compaction_logging import (
        CompactionEvent,
        llm_call_count_for_strategy,
        subagent_trajectory_refs_to_dicts,
    )

    self._nel_ensure_compaction_state()
    self._nel_compaction_count += 1
    compaction_index = self._nel_compaction_count
    return CompactionEvent(
        compaction_index=compaction_index,
        trigger=trigger,
        strategy=strategy,
        replacement_content=replacement_content,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_intermediate=tokens_intermediate,
        llm_call_count=llm_call_count_for_strategy(strategy),
        mechanism=self._nel_compaction_mechanism(),
        subagent_trajectory_ref=subagent_trajectory_refs_to_dicts(subagent_refs),
        handoff_prompt=handoff_prompt,
        attempts=attempts or None,
    )


def _terminus2_nel_record_failed_compaction_attempt(
    self,
    attempts: list[CompactionAttempt],
    strategy: CompactionStrategy,
    error: str,
) -> None:
    from nemo_evaluator.solvers.compaction_logging import CompactionAttempt, llm_call_count_for_strategy

    attempts.append(
        CompactionAttempt(
            strategy=strategy,
            error=error,
            llm_calls=llm_call_count_for_strategy(strategy),
        )
    )


def _terminus2_nel_set_proactive_pending_compaction(self, chat, prompt: str, subagent_refs: Any) -> None:
    from nemo_evaluator.solvers.compaction_logging import serialize_chat_messages

    self._nel_ensure_compaction_state()
    tokens_before = self._nel_stashed_tokens_before
    tokens_after = self._nel_stashed_tokens_after
    tokens_intermediate = self._nel_stashed_tokens_intermediate
    if tokens_before is None or tokens_after is None:
        raise RuntimeError(
            "Proactive compaction token snapshots are missing; "
            "_check_proactive_summarization was not patched to capture them."
        )
    self._nel_last_unwind_pairs = 0
    self._nel_last_unwind_target = None
    self._nel_cle_chat_reset_applied = False
    replacement_messages = list(chat.messages) + [{"role": "user", "content": prompt}]
    self._nel_pending_compaction = self._nel_make_compaction_event(
        trigger="terminus_proactive_threshold",
        strategy="terminus_three_phase_subagent",
        replacement_content=serialize_chat_messages(replacement_messages),
        subagent_refs=subagent_refs,
        attempts=None,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_intermediate=tokens_intermediate,
        handoff_prompt=prompt,
    )
    self._nel_stashed_tokens_before = None
    self._nel_stashed_tokens_after = None
    self._nel_stashed_tokens_intermediate = None


def _terminus2_nel_set_cle_pending_compaction(
    self,
    chat,
    summary_prompt: str,
    strategy: CompactionStrategy,
    attempts: list[CompactionAttempt],
    subagent_refs: Any,
    tokens_before: CompactionTokens,
    tokens_after: CompactionTokens,
    tokens_after_unwind: CompactionTokens | None,
    tokens_after_chat_reset: CompactionTokens | None,
) -> None:
    from nemo_evaluator.solvers.compaction_logging import (
        CompactionTokensIntermediate,
        serialize_chat_messages,
    )

    self._nel_ensure_compaction_state()
    tokens_intermediate = None
    if tokens_after_unwind is not None or tokens_after_chat_reset is not None:
        tokens_intermediate = CompactionTokensIntermediate(
            after_unwind=tokens_after_unwind,
            after_chat_reset=tokens_after_chat_reset,
        )
    self._nel_pending_compaction = self._nel_make_compaction_event(
        trigger="terminus_context_length_exceeded",
        strategy=strategy,
        replacement_content=serialize_chat_messages(list(chat.messages)),
        subagent_refs=subagent_refs,
        attempts=attempts or None,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        tokens_intermediate=tokens_intermediate,
        handoff_prompt=summary_prompt,
    )


def _terminus2_nel_flush_pending_compaction(self, chat) -> None:
    from nemo_evaluator.solvers.compaction_logging import compaction_event_to_harbor_step

    self._nel_ensure_compaction_state()
    event = self._nel_pending_compaction
    if event is None:
        return

    step_id = len(self._trajectory_steps) + 1
    step = compaction_event_to_harbor_step(event, step_id)
    self._trajectory_steps.append(step)

    handoff_prompt = event.handoff_prompt
    self._nel_pending_compaction = None
    self._pending_subagent_refs = None
    self._pending_handoff_prompt = None
    self._nel_cle_chat_reset_applied = False
    self._nel_last_unwind_pairs = 0
    self._nel_last_unwind_target = None

    if self._linear_history and handoff_prompt:
        self._split_trajectory_on_summarization(handoff_prompt)


def _terminus2_nel_dump_trajectory_with_continuation_index(self, continuation_index: int) -> None:
    from harbor.models.trajectories.agent import Agent
    from harbor.models.trajectories.final_metrics import FinalMetrics
    from harbor.models.trajectories.trajectory import Trajectory
    from harbor.utils.trajectory_utils import format_trajectory_json

    if not self._context:
        self.logger.warning("No context available, skipping trajectory dump")
        return

    final_metrics = FinalMetrics(
        total_prompt_tokens=self._context.n_input_tokens,
        total_completion_tokens=self._context.n_output_tokens,
        total_cached_tokens=self._context.n_cache_tokens or 0,
        total_cost_usd=self._context.cost_usd if self._context.cost_usd else None,
    )

    agent_extra = {
        "parser": self._parser_name,
        "temperature": self._temperature,
    }
    if self._llm_kwargs:
        agent_extra["llm_kwargs"] = self._llm_kwargs
    if self._linear_history and continuation_index > 0:
        agent_extra["continuation_index"] = continuation_index

    continued_trajectory_ref = None
    if self._linear_history and continuation_index < self._summarization_count:
        next_continuation_index = continuation_index + 1
        continued_trajectory_ref = f"trajectory.cont-{next_continuation_index}.json"

    trajectory = Trajectory(
        session_id=self._session_id,
        agent=Agent(
            name=self.name(),
            version=self.version() or "unknown",
            model_name=self._model_name,
            extra=agent_extra,
        ),
        steps=self._trajectory_steps,
        final_metrics=final_metrics,
        continued_trajectory_ref=continued_trajectory_ref,
    )

    if self._linear_history and continuation_index > 0:
        trajectory_path = self.logs_dir / f"trajectory.cont-{continuation_index}.json"
    else:
        trajectory_path = self.logs_dir / "trajectory.json"

    trajectory_dict = trajectory.to_json_dict()

    try:
        with open(trajectory_path, "w") as f:
            json_str = format_trajectory_json(trajectory_dict)
            f.write(json_str)
        self.logger.debug(f"Trajectory dumped to {trajectory_path}")
    except Exception as e:
        self.logger.error(f"Failed to dump trajectory: {e}")


def _apply_source_replacements(src: str, replacements: list[tuple[str, str]], *, label: str) -> str:
    for anchor, replacement in replacements:
        occurrences = src.count(anchor)
        if occurrences != 1:
            raise RuntimeError(
                f"Cannot patch {label}: expected exactly one match for an anchor but found "
                f"{occurrences}. Harbor's terminus_2.py has diverged from the expected source."
            )
        src = src.replace(anchor, replacement, 1)
    return src


def _strip_def_indent(src: str) -> str:
    lines = src.splitlines(True)
    if not lines:
        return src
    first = lines[0]
    indent = len(first) - len(first.lstrip(" "))
    if indent == 0:
        return src
    prefix = " " * indent
    return "".join(line[indent:] if line.startswith(prefix) else line for line in lines)


_TERMINUS_COMPACTION_RUN_LOOP_REPLACEMENTS = [
    (
        "                if proactive_summary_result:\n"
        "                    prompt, subagent_refs = proactive_summary_result\n"
        "                    # Store subagent_refs to add a system step later\n"
        "                    self._pending_subagent_refs = subagent_refs\n"
        "                    # Also store the handoff prompt to add as a user step\n"
        "                    self._pending_handoff_prompt = prompt\n",
        "                if proactive_summary_result:\n"
        "                    prompt, subagent_refs = proactive_summary_result\n"
        "                    self._pending_subagent_refs = subagent_refs\n"
        "                    self._pending_handoff_prompt = prompt\n"
        "                    self._nel_set_proactive_pending_compaction(chat, prompt, subagent_refs)\n",
    ),
    (
        "            # If we have pending subagent refs, add a system step to record the delegation\n"
        "            # This must happen before we build the agent step\n"
        "            # We use len(self._trajectory_steps) + 1 as the step_id to ensure it's sequential\n"
        "            if self._pending_subagent_refs:\n"
        "                self._trajectory_steps.append(\n"
        "                    Step(\n"
        "                        step_id=len(self._trajectory_steps) + 1,\n"
        "                        timestamp=datetime.now(timezone.utc).isoformat(),\n"
        '                        source="system",\n'
        '                        message="Performed context summarization and handoff to continue task.",\n'
        "                        observation=Observation(\n"
        "                            results=[\n"
        "                                ObservationResult(\n"
        "                                    subagent_trajectory_ref=self._pending_subagent_refs\n"
        "                                )\n"
        "                            ]\n"
        "                        ),\n"
        "                    )\n"
        "                )\n"
        "                self._pending_subagent_refs = None\n"
        "\n"
        "            # Handle handoff prompt based on linear_history mode\n"
        "            if self._pending_handoff_prompt:\n"
        "                # If linear_history mode is enabled, split trajectory immediately WITHOUT adding handoff step\n"
        "                # The handoff step will be added to the continuation trajectory during the split\n"
        "                if self._linear_history:\n"
        "                    self._split_trajectory_on_summarization(\n"
        "                        self._pending_handoff_prompt\n"
        "                    )\n"
        "                else:\n"
        "                    # For non-linear mode, add the handoff prompt as a user step\n"
        "                    self._trajectory_steps.append(\n"
        "                        Step(\n"
        "                            step_id=len(self._trajectory_steps) + 1,\n"
        "                            timestamp=datetime.now(timezone.utc).isoformat(),\n"
        '                            source="user",\n'
        "                            message=self._pending_handoff_prompt,\n"
        "                        )\n"
        "                    )\n"
        "                self._pending_handoff_prompt = None\n",
        "            self._nel_flush_pending_compaction(chat)\n",
    ),
]

_TERMINUS_COMPACTION_PROACTIVE_CHECK_REPLACEMENTS = [
    (
        "            try:\n"
        "                summary_prompt, subagent_trajectory_refs = await self._summarize(\n"
        "                    chat, original_instruction, session\n"
        "                )\n"
        "                return (summary_prompt, subagent_trajectory_refs)\n",
        "            try:\n"
        "                nel_proactive_tokens_before = self._nel_token_snapshot(chat)\n"
        "                summary_prompt, subagent_trajectory_refs = await self._summarize(\n"
        "                    chat, original_instruction, session\n"
        "                )\n"
        "                nel_proactive_tokens_after = self._nel_token_snapshot(chat)\n"
        "                self._nel_stash_compaction_token_snapshots(\n"
        "                    nel_proactive_tokens_before,\n"
        "                    nel_proactive_tokens_after,\n"
        "                    None,\n"
        "                )\n"
        "                return (summary_prompt, subagent_trajectory_refs)\n",
    ),
]

_TERMINUS_COMPACTION_SHORT_SUMMARY_REPLACEMENTS = [
    (
        "                    short_llm_response: LLMResponse = await self._llm.call(\n"
        "                        prompt=short_prompt,\n"
        "                        **self._llm_call_kwargs,\n"
        "                    )\n",
        "                    short_llm_response: LLMResponse = await self._llm.call(\n"
        "                        prompt=short_prompt,\n"
        "                        **self._nel_compaction_llm_call_kwargs(),\n"
        "                    )\n",
    ),
]

_TERMINUS_COMPACTION_QUERY_LLM_REPLACEMENTS = [
    (
        "            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)\n"
        "\n"
        "            summary_prompt = None\n",
        "            nel_cle_tokens_before = self._nel_token_snapshot(chat)\n"
        "            nel_cle_tokens_after_unwind = None\n"
        "            nel_cle_tokens_after_chat_reset = None\n"
        "            self._unwind_messages_to_free_tokens(chat, target_free_tokens=4000)\n"
        "            nel_cle_tokens_after_unwind = self._nel_token_snapshot(chat)\n"
        "            nel_cle_attempts = []\n"
        '            nel_cle_strategy = "terminus_ultimate_fallback"\n'
        "            nel_cle_subagent_refs = None\n"
        "\n"
        "            summary_prompt = None\n",
    ),
    (
        '                self.logger.debug("SUMMARIZATION: Full summary succeeded")\n',
        '                self.logger.debug("SUMMARIZATION: Full summary succeeded")\n'
        '                nel_cle_strategy = "terminus_three_phase_subagent"\n'
        "                nel_cle_subagent_refs = subagent_trajectory_refs\n",
    ),
    (
        '                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")\n',
        '                self.logger.debug(f"SUMMARIZATION: Full summary failed: {e}")\n'
        "                self._nel_record_failed_compaction_attempt(\n"
        '                    nel_cle_attempts, "terminus_three_phase_subagent", str(e)\n'
        "                )\n",
    ),
    (
        '                    self.logger.debug("SUMMARIZATION: Short summary succeeded")\n',
        '                    self.logger.debug("SUMMARIZATION: Short summary succeeded")\n'
        '                    nel_cle_strategy = "terminus_short_fallback"\n',
    ),
    (
        '                    self.logger.error(f"SUMMARIZATION: Short summary failed: {e}")\n',
        '                    self.logger.error(f"SUMMARIZATION: Short summary failed: {e}")\n'
        "                    self._nel_record_failed_compaction_attempt(\n"
        '                        nel_cle_attempts, "terminus_short_fallback", str(e)\n'
        "                    )\n",
    ),
    *_TERMINUS_COMPACTION_SHORT_SUMMARY_REPLACEMENTS,
]

_TERMINUS_COMPACTION_QUERY_LLM_CLE_CHAT_RESET_REPLACEMENT = (
    "            if full_summarize_failed_with_cle:\n"
    "                chat._messages = [chat.messages[0]]\n"
    "                chat.reset_response_chain()\n",
    "            if full_summarize_failed_with_cle:\n"
    "                chat._messages = [chat.messages[0]]\n"
    "                chat.reset_response_chain()\n"
    "                self._nel_cle_chat_reset_applied = True\n"
    "                nel_cle_tokens_after_chat_reset = self._nel_token_snapshot(chat)\n",
)

_TERMINUS_COMPACTION_QUERY_LLM_NO_CLE_CHAT_RESET_REPLACEMENT = (
    "            if prompt_path is not None:\n"
    "                prompt_path.write_text(summary_prompt)\n"
    "\n"
    "            try:\n"
    "                start_time = time.time()\n"
    "                llm_response = await chat.chat(\n",
    "            if prompt_path is not None:\n"
    "                prompt_path.write_text(summary_prompt)\n"
    "\n"
    "            self._nel_cle_chat_reset_applied = False\n"
    "\n"
    "            try:\n"
    "                start_time = time.time()\n"
    "                llm_response = await chat.chat(\n",
)

_TERMINUS_COMPACTION_QUERY_LLM_PENDING_FLUSH_REPLACEMENT = (
    "            if response_path is not None:\n"
    "                response_path.write_text(llm_response.content)\n"
    "            return llm_response\n"
    "\n"
    "        except OutputLengthExceededError as e:\n",
    "            if response_path is not None:\n"
    "                response_path.write_text(llm_response.content)\n"
    "            nel_cle_tokens_after = self._nel_token_snapshot(chat)\n"
    "            self._nel_set_cle_pending_compaction(\n"
    "                chat,\n"
    "                summary_prompt,\n"
    "                nel_cle_strategy,\n"
    "                nel_cle_attempts,\n"
    "                nel_cle_subagent_refs,\n"
    "                nel_cle_tokens_before,\n"
    "                nel_cle_tokens_after,\n"
    "                nel_cle_tokens_after_unwind,\n"
    "                nel_cle_tokens_after_chat_reset,\n"
    "            )\n"
    "            return llm_response\n"
    "\n"
    "        except OutputLengthExceededError as e:\n",
)


def _terminus2_nel_compaction_llm_call_kwargs(self) -> dict[str, Any]:
    from nemo_evaluator.adapters.call_kind import merge_compaction_extra_headers

    return merge_compaction_extra_headers(self._llm_call_kwargs)


def _patch_terminus_compaction_logging() -> None:
    global _TERMINUS_COMPACTION_PATCHED, _TERMINUS_QUERY_LLM_SRC
    if _TERMINUS_COMPACTION_PATCHED:
        return

    import inspect
    import textwrap

    from harbor.agents.terminus_2 import terminus_2 as terminus2_mod

    terminus2_mod.Terminus2._nel_ensure_compaction_state = _terminus2_nel_ensure_compaction_state
    terminus2_mod.Terminus2._nel_stash_compaction_token_snapshots = _terminus2_nel_stash_compaction_token_snapshots
    terminus2_mod.Terminus2._nel_token_snapshot = _terminus2_nel_token_snapshot
    terminus2_mod.Terminus2._nel_compaction_mechanism = _terminus2_nel_compaction_mechanism
    terminus2_mod.Terminus2._nel_make_compaction_event = _terminus2_nel_make_compaction_event
    terminus2_mod.Terminus2._nel_record_failed_compaction_attempt = _terminus2_nel_record_failed_compaction_attempt
    terminus2_mod.Terminus2._nel_set_proactive_pending_compaction = _terminus2_nel_set_proactive_pending_compaction
    terminus2_mod.Terminus2._nel_set_cle_pending_compaction = _terminus2_nel_set_cle_pending_compaction
    terminus2_mod.Terminus2._nel_flush_pending_compaction = _terminus2_nel_flush_pending_compaction
    terminus2_mod.Terminus2._nel_compaction_llm_call_kwargs = _terminus2_nel_compaction_llm_call_kwargs
    terminus2_mod.Terminus2._dump_trajectory_with_continuation_index = (
        _terminus2_nel_dump_trajectory_with_continuation_index
    )

    run_loop_src = inspect.getsource(inspect.unwrap(terminus2_mod.Terminus2._run_agent_loop))
    if "_nel_flush_pending_compaction" in run_loop_src:
        logger.info("Terminus2._run_agent_loop already contains compaction logging; skipping run-loop patch")
    else:
        run_loop_src = _apply_source_replacements(
            run_loop_src,
            _TERMINUS_COMPACTION_RUN_LOOP_REPLACEMENTS,
            label="Terminus2._run_agent_loop for compaction logging",
        )
        run_loop_namespace: dict[str, Any] = {}
        exec(textwrap.dedent(run_loop_src), terminus2_mod.__dict__, run_loop_namespace)
        terminus2_mod.Terminus2._run_agent_loop = run_loop_namespace["_run_agent_loop"]

        proactive_method = inspect.unwrap(terminus2_mod.Terminus2._check_proactive_summarization)
        try:
            proactive_src = inspect.getsource(proactive_method)
        except OSError:
            proactive_src = ""
        if "_nel_stash_compaction_token_snapshots" not in proactive_src:
            if not proactive_src:
                if "_nel_stash_compaction_token_snapshots" not in proactive_method.__code__.co_names:
                    raise RuntimeError(
                        "Cannot patch Terminus2._check_proactive_summarization for compaction logging: "
                        "source is unavailable and the method is not already patched."
                    )
            else:
                proactive_src = _apply_source_replacements(
                    proactive_src,
                    _TERMINUS_COMPACTION_PROACTIVE_CHECK_REPLACEMENTS,
                    label="Terminus2._check_proactive_summarization for compaction logging",
                )
                proactive_namespace: dict[str, Any] = {}
                exec(textwrap.dedent(proactive_src), terminus2_mod.__dict__, proactive_namespace)
                terminus2_mod.Terminus2._check_proactive_summarization = proactive_namespace[
                    "_check_proactive_summarization"
                ]

    if _TERMINUS_QUERY_LLM_SRC is None:
        try:
            _TERMINUS_QUERY_LLM_SRC = inspect.getsource(inspect.unwrap(terminus2_mod.Terminus2._query_llm))
        except OSError:
            _TERMINUS_QUERY_LLM_SRC = ""
    query_src = _TERMINUS_QUERY_LLM_SRC
    if query_src and "_nel_set_cle_pending_compaction" not in query_src:
        query_replacements = list(_TERMINUS_COMPACTION_QUERY_LLM_REPLACEMENTS)
        if "full_summarize_failed_with_cle" in query_src:
            query_replacements.append(_TERMINUS_COMPACTION_QUERY_LLM_CLE_CHAT_RESET_REPLACEMENT)
        else:
            query_replacements.append(_TERMINUS_COMPACTION_QUERY_LLM_NO_CLE_CHAT_RESET_REPLACEMENT)
        query_replacements.append(_TERMINUS_COMPACTION_QUERY_LLM_PENDING_FLUSH_REPLACEMENT)
        query_src = _apply_source_replacements(
            query_src,
            query_replacements,
            label="Terminus2._query_llm for compaction logging",
        )
        query_namespace: dict[str, Any] = {}
        exec(textwrap.dedent(query_src), terminus2_mod.__dict__, query_namespace)
        terminus2_mod.Terminus2._query_llm = query_namespace["_query_llm"]
        _TERMINUS_QUERY_LLM_SRC = query_src
    elif query_src and "_nel_compaction_llm_call_kwargs()" not in query_src:
        short_summary_anchor = _TERMINUS_COMPACTION_SHORT_SUMMARY_REPLACEMENTS[0][0]
        if short_summary_anchor not in query_src:
            raise RuntimeError(
                "Cannot patch Terminus2._query_llm short-summary call for compaction turn marker: "
                "expected source anchor not found."
            )
        query_src = _apply_source_replacements(
            query_src,
            _TERMINUS_COMPACTION_SHORT_SUMMARY_REPLACEMENTS,
            label="Terminus2._query_llm short-summary compaction turn marker",
        )
        query_namespace = {}
        exec(textwrap.dedent(query_src), terminus2_mod.__dict__, query_namespace)
        terminus2_mod.Terminus2._query_llm = query_namespace["_query_llm"]
        _TERMINUS_QUERY_LLM_SRC = query_src

    _TERMINUS_COMPACTION_PATCHED = True
    logger.info("Patched Terminus2 for ATIF context compaction logging")


_TERMINUS_COMPACTION_TURN_MARKER_PATCHED = False

_TERMINUS_COMPACTION_RUN_SUBAGENT_REPLACEMENTS = [
    (
        "        summary_text: str,\n"
        '        subagent_name_for_logging: str = "subagent",\n'
        "    ) -> tuple[LLMResponse, SubagentTrajectoryRef]:\n",
        "        summary_text: str,\n"
        '        subagent_name_for_logging: str = "subagent",\n'
        "        llm_call_kwargs: dict | None = None,\n"
        "    ) -> tuple[LLMResponse, SubagentTrajectoryRef]:\n",
    ),
    (
        "        response: LLMResponse = await self._llm.call(\n"
        "            prompt=prompt,\n"
        "            message_history=message_history,\n"
        "            **self._llm_call_kwargs,\n"
        "        )\n",
        "        response: LLMResponse = await self._llm.call(\n"
        "            prompt=prompt,\n"
        "            message_history=message_history,\n"
        "            **(llm_call_kwargs if llm_call_kwargs is not None else self._llm_call_kwargs),\n"
        "        )\n",
    ),
]

_TERMINUS_COMPACTION_SUMMARIZE_REPLACEMENTS = [
    (
        '            subagent_name_for_logging="summary generation LLM call",\n        )\n',
        '            subagent_name_for_logging="summary generation LLM call",\n'
        "            llm_call_kwargs=self._nel_compaction_llm_call_kwargs(),\n"
        "        )\n",
    ),
    (
        '            subagent_name_for_logging="questions subagent",\n        )\n',
        '            subagent_name_for_logging="questions subagent",\n'
        "            llm_call_kwargs=self._nel_compaction_llm_call_kwargs(),\n"
        "        )\n",
    ),
    (
        '            subagent_name_for_logging="answers subagent",\n        )\n',
        '            subagent_name_for_logging="answers subagent",\n'
        "            llm_call_kwargs=self._nel_compaction_llm_call_kwargs(),\n"
        "        )\n",
    ),
]


def _patch_terminus_compaction_turn_marker() -> None:
    global _TERMINUS_COMPACTION_TURN_MARKER_PATCHED
    if _TERMINUS_COMPACTION_TURN_MARKER_PATCHED:
        return

    import inspect

    from harbor.agents.terminus_2 import terminus_2 as terminus2_mod

    terminus2_mod.Terminus2._nel_compaction_llm_call_kwargs = _terminus2_nel_compaction_llm_call_kwargs

    run_subagent_src = inspect.getsource(inspect.unwrap(terminus2_mod.Terminus2._run_subagent))
    if "llm_call_kwargs if llm_call_kwargs is not None" not in run_subagent_src:
        run_subagent_src = _apply_source_replacements(
            run_subagent_src,
            _TERMINUS_COMPACTION_RUN_SUBAGENT_REPLACEMENTS,
            label="Terminus2._run_subagent for optional llm_call_kwargs",
        )
        run_subagent_namespace: dict[str, Any] = {}
        exec(_strip_def_indent(run_subagent_src), terminus2_mod.__dict__, run_subagent_namespace)
        terminus2_mod.Terminus2._run_subagent = run_subagent_namespace["_run_subagent"]

    summarize_method = inspect.unwrap(terminus2_mod.Terminus2._summarize)
    try:
        summarize_src = inspect.getsource(summarize_method)
    except OSError:
        summarize_src = ""
    if summarize_src and "_nel_compaction_llm_call_kwargs()" not in summarize_src:
        summarize_src = _apply_source_replacements(
            summarize_src,
            _TERMINUS_COMPACTION_SUMMARIZE_REPLACEMENTS,
            label="Terminus2._summarize compaction llm_call_kwargs",
        )
        summarize_namespace: dict[str, Any] = {}
        exec(_strip_def_indent(summarize_src), terminus2_mod.__dict__, summarize_namespace)
        terminus2_mod.Terminus2._summarize = summarize_namespace["_summarize"]

    _TERMINUS_COMPACTION_TURN_MARKER_PATCHED = True
    logger.info("Patched Terminus2 to mark compaction LLM calls for turn_counter")


_TOKENIZER_REGISTRY: dict[str, dict] = {}
_TOKEN_COUNTER_PATCHED = False
_MISSING_TOKENIZER_WARNED: set[str] = set()
_IGNORED_TOKENIZER_LOGGED: set[str] = set()


def _build_custom_tokenizer(spec: str) -> dict:
    from litellm.utils import create_pretrained_tokenizer, create_tokenizer

    spec_path = Path(spec)
    if spec_path.is_file():
        return create_tokenizer(spec_path.read_text())
    return create_pretrained_tokenizer(spec)


def _install_token_counter_patch() -> None:
    global _TOKEN_COUNTER_PATCHED
    if _TOKEN_COUNTER_PATCHED:
        return

    import litellm.utils as litellm_utils

    _original_token_counter = litellm_utils.token_counter

    def _patched_token_counter(*args, **kwargs):
        model = kwargs["model"] if "model" in kwargs else (args[0] if args else "")
        passes_custom = kwargs.get("custom_tokenizer") is not None or len(args) >= 2
        if model in _TOKENIZER_REGISTRY and not passes_custom:
            kwargs["custom_tokenizer"] = _TOKENIZER_REGISTRY[model]
        return _original_token_counter(*args, **kwargs)

    litellm_utils.token_counter = _patched_token_counter
    _TOKEN_COUNTER_PATCHED = True


def _register_harbor_tokenizer(model_name: str, spec: str) -> None:
    if model_name not in _TOKENIZER_REGISTRY:
        try:
            tokenizer = _build_custom_tokenizer(spec)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load tokenizer {spec!r} configured for model {model_name!r}. "
                "Provide a valid Hugging Face repo id or a local tokenizer.json path, or "
                "remove the 'tokenizer' field from the model service config. "
                f"Underlying error: {exc}"
            ) from exc
        _TOKENIZER_REGISTRY[model_name] = tokenizer
        logger.info("Registered custom tokenizer %r for model %r", spec, model_name)
    _install_token_counter_patch()


def _configure_terminus_tokenizer(*, is_terminus2: bool, model_name: str, tokenizer: str | None) -> None:
    if not is_terminus2:
        if tokenizer and model_name and model_name not in _IGNORED_TOKENIZER_LOGGED:
            _IGNORED_TOKENIZER_LOGGED.add(model_name)
            logger.info(
                "Tokenizer %r is configured for model %r but is only used by the "
                "terminus-2 agent; ignoring it for the current agent.",
                tokenizer,
                model_name,
            )
        return

    if not tokenizer:
        if model_name and model_name not in _MISSING_TOKENIZER_WARNED:
            _MISSING_TOKENIZER_WARNED.add(model_name)
            logger.warning(
                "No tokenizer configured for model %r used with terminus-2; token "
                "counts will use litellm's tokenizer and may be inaccurate. Set the "
                "'tokenizer' field (Hugging Face repo id or local tokenizer.json path) on "
                "the model service config.",
                model_name,
            )
        return

    if model_name:
        _register_harbor_tokenizer(model_name, tokenizer)


def _count_zero_token_agent_turns(trajectory: Any) -> int:
    """Count agent LLM turns recorded with zero prompt+completion tokens.

    A zero-token turn means an LLM response carried no usage — the symptom of a
    context reset/condensation that failed to fire (local ``litellm.token_counter``
    vs. API token-count mismatch). Surfaced for observability, not classified as
    an infra error.
    """
    docs = trajectory if isinstance(trajectory, list) else [trajectory]
    count = 0
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        for step in doc.get("steps") or []:
            if not isinstance(step, dict) or step.get("source") != "agent":
                continue
            metrics = step.get("metrics")
            if (
                isinstance(metrics, dict)
                and metrics.get("prompt_tokens") == 0
                and metrics.get("completion_tokens") == 0
            ):
                count += 1
    return count


class HarborSolver:
    """Runs any Harbor agent inside an evaluator :class:`Sandbox`.

    Agent resolution (``harbor_agent`` parameter):
      - Built-in name (e.g. ``"openhands"``)
      - Import path  (e.g. ``"my_pkg:MyAgent"``)
      - Directory path (e.g. ``"./agents/my-agent/"``)
    """

    def __init__(
        self,
        *,
        harbor_agent: str,
        harbor_agent_kwargs: dict[str, Any] | None = None,
        model_url: str = "",
        model_id: str = "",
        timeout: float = 1800.0,
        run_timeout: float | None = None,
        api_key: str | None = None,
        container_env: dict[str, str] | None = None,
        max_input_tokens: int | None = None,
        max_output_tokens: int | None = None,
        tokenizer: str | None = None,
        cmd_timeout: float | None = None,
        timeout_strategy: str = "override",
        max_agent_timeout: float | None = None,
        skill: str | None = None,
        skill_dir: str | None = None,
    ) -> None:
        _check_harbor_installed()
        self._skill = skill
        self._skill_dir = Path(skill_dir) if skill_dir else None
        self._harbor_agent = harbor_agent
        self._harbor_agent_kwargs = harbor_agent_kwargs or {}
        self._model_url = model_url
        self._model_id = model_id
        self._timeout = timeout
        self._run_timeout = run_timeout or timeout
        self._cmd_timeout = cmd_timeout
        self._timeout_strategy = timeout_strategy
        self._max_agent_timeout = max_agent_timeout
        self._container_env = dict(container_env or {})
        self._container_env.setdefault("PIP_INDEX_URL", "https://pypi.org/simple")
        self._container_env.setdefault("LITELLM_LOG", "ERROR")
        self._container_env.setdefault("LITELLM_TELEMETRY", "false")
        if harbor_agent.lower() in ("openhands", "openhands-sdk"):
            self._container_env.setdefault("OH_PRELOAD_TOOLS", "false")
            self._container_env.setdefault("SECURITY_CONFIRMATION_MODE", "false")
            self._container_env.setdefault("SECURITY_ENABLE_SECURITY_ANALYZER", "false")
            self._container_env.setdefault("LLM_NATIVE_TOOL_CALLING", "true")
            llm_kw = self._harbor_agent_kwargs.get("llm_kwargs")
            if isinstance(llm_kw, dict):
                llm_timeout = llm_kw.get("timeout")
                if llm_timeout is not None:
                    self._container_env.setdefault("LLM_TIMEOUT", str(int(float(llm_timeout))))
            # Config-driven OpenHands condenser: agent_kwargs.condenser_max_size ->
            # env var read by the run_agent.py patch (see _reasoning_script).
            cms = self._harbor_agent_kwargs.get("condenser_max_size")
            if cms is not None:
                self._container_env.setdefault("OH_CONDENSER_MAX_SIZE", str(int(cms)))
            cmt = self._harbor_agent_kwargs.get("condenser_max_tokens")
            if cmt is not None:
                self._container_env.setdefault("OH_CONDENSER_MAX_TOKENS", str(int(cmt)))
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
        self._tokenizer = tokenizer
        self._api_key = _resolve_api_key(api_key)
        if not self._api_key:
            self._api_key = "no-key-needed"
            logger.info("No API key found — using dummy key for self-hosted model")

        self._container_env.setdefault("LLM_API_KEY", self._api_key)
        if model_id:
            self._container_env.setdefault(
                "LLM_MODEL",
                _model_id_for_openai(model_id, bool(model_url), agent=harbor_agent),
            )

        if harbor_agent.lower() == "claude-code":
            _ensure_claude_host_env(self._api_key, model_url)
        else:
            _ensure_host_env(self._api_key, self._model_id, has_custom_url=bool(model_url))

    def _create_agent(self, logs_dir: Path, *, model_url: str = "") -> Any:
        from harbor.agents.factory import AgentFactory

        # AgentFactory eagerly imports every built-in agent (including
        # terminus_2), so the byte-length patch target is
        # loaded regardless of which agent we build.
        _patch_terminus_tmux_send_keys()

        kwargs = dict(self._harbor_agent_kwargs)
        # Consumed via OH_CONDENSER_MAX_SIZE env (see __init__); not a harbor agent kwarg.
        kwargs.pop("condenser_max_size", None)
        kwargs.pop("condenser_max_tokens", None)
        url = model_url or self._model_url
        model_id = _model_id_for_openai(self._model_id, bool(url), agent=self._harbor_agent) if self._model_id else ""
        is_terminus2 = self._harbor_agent.replace("_", "-").lower() == "terminus-2"
        if is_terminus2:
            _patch_terminus_cle_reset()
            _patch_terminus_unwind_min_pairs()
            _patch_terminus_api_token_anchor()
            _patch_harbor_lite_llm_context_length_matcher()
            _patch_terminus_compaction_logging()
            _patch_terminus_compaction_turn_marker()
        if "model_name" not in kwargs and model_id:
            kwargs["model_name"] = model_id
        _configure_terminus_tokenizer(
            is_terminus2=is_terminus2,
            model_name=kwargs.get("model_name") or model_id,
            tokenizer=self._tokenizer,
        )
        if "api_base" not in kwargs and url:
            kwargs["api_base"] = url
        if "api_key" not in kwargs and self._api_key:
            kwargs["api_key"] = self._api_key
        if self._api_key:
            llm_kw = kwargs.get("llm_kwargs")
            if llm_kw is None:
                kwargs["llm_kwargs"] = {"api_key": self._api_key}
            elif isinstance(llm_kw, dict):
                llm_kw.setdefault("api_key", self._api_key)

        if "model_info" not in kwargs:
            kwargs["model_info"] = {
                "max_input_tokens": self._max_input_tokens or 262144,
                "max_output_tokens": self._max_output_tokens or 262144,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }

        name = self._harbor_agent
        if ":" in name:
            return AgentFactory.create_agent_from_import_path(name, logs_dir=logs_dir, **kwargs)
        agent_path = Path(name)
        if agent_path.is_dir():
            from nemo_evaluator.solvers.byob_agent import ByobInstalledAgent

            return ByobInstalledAgent(agent_dir=agent_path, logs_dir=logs_dir, **kwargs)
        return AgentFactory.create_agent_from_name(name, logs_dir=logs_dir, **kwargs)

    async def _wait_for_agent(
        self,
        agent_task: asyncio.Task[None],
        sandbox: "Sandbox",
        agent_started_at: float,
        effective_timeout: float,
        jitter: float,
    ) -> tuple[bool, Exception | None]:
        """Run *agent_task* with a two-phase timeout.

        Phase 1 (short): wait ``min(600, 15% of run_timeout)`` seconds.
            If the agent has produced no log files, abort early with
            `GracefulError` — the model is likely unreachable.
        Phase 2 (remainder): wait the rest of ``effective_timeout``.

        Returns ``(timed_out, agent_error)`` — *timed_out* is True when
        the agent didn't finish in time, *agent_error* captures any
        exception the agent raised (instead of propagating it) so that
        ``solve()`` can still collect partial results.
        Raises `GracefulError` if no progress was detected.
        """
        resolved_timeout = effective_timeout - jitter
        no_progress_timeout = min(600.0, resolved_timeout * 0.15)

        # Phase 1 — wait for first sign of life
        done, _ = await asyncio.wait(
            {agent_task},
            timeout=no_progress_timeout + jitter,
        )

        if not done:
            # Agent still running — probe sandbox for log output
            has_progress = True
            if sandbox.is_running:
                try:
                    probe = await sandbox.exec(
                        "find /logs/agent -type f 2>/dev/null | wc -l",
                        timeout_sec=15,
                    )
                    count = int(probe.stdout.strip()) if probe.stdout else 0
                    has_progress = count > 0
                except Exception:
                    pass  # can't probe → assume progress

            if not has_progress:
                logger.warning(
                    "HarborSolver: no agent progress after %.0fs — aborting early (model may be unreachable)",
                    no_progress_timeout,
                )
                agent_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await agent_task
                raise InfraError(
                    f"Agent made no progress after {no_progress_timeout:.0f}s "
                    f"(run_timeout={self._run_timeout:.0f}s). "
                    "Model endpoint may be unreachable or overloaded."
                )

            # Phase 2 — progress confirmed, wait remaining time
            remaining = effective_timeout - (time.monotonic() - agent_started_at)
            done, _ = await asyncio.wait(
                {agent_task},
                timeout=max(0.0, remaining),
            )

        # If agent still running after both phases → timed out
        timed_out = agent_task not in done
        if timed_out:
            logger.warning(
                "HarborSolver: agent.run() timed out after %.0fs "
                "(effective=%.0fs+%.0fs jitter, strategy=%s) — collecting partial results",
                time.monotonic() - agent_started_at,
                effective_timeout - jitter,
                jitter,
                self._timeout_strategy,
            )
            if sandbox.is_running:
                try:
                    await sandbox.exec(
                        "python3 - <<'PY'\n"
                        "import os, signal, subprocess\n"
                        "me = os.getpid()\n"
                        "try:\n"
                        "    out = subprocess.check_output(['ps', '-eo', 'pid=,comm=,args='], text=True)\n"
                        "except Exception:\n"
                        "    out = ''\n"
                        "for line in out.splitlines():\n"
                        "    parts = line.strip().split(None, 2)\n"
                        "    if len(parts) < 3:\n"
                        "        continue\n"
                        "    try:\n"
                        "        pid = int(parts[0])\n"
                        "    except ValueError:\n"
                        "        continue\n"
                        "    comm, args = parts[1], parts[2]\n"
                        "    if pid != me and 'python' in comm and '/installed-agent/run_agent.py' in args:\n"
                        "        os.kill(pid, signal.SIGTERM)\n"
                        "PY",
                        timeout_sec=10,
                    )
                    done_after_signal, _ = await asyncio.wait({agent_task}, timeout=30.0)
                    if agent_task in done_after_signal:
                        logger.info("HarborSolver: timed-out agent flushed after SIGTERM")
                except Exception:
                    logger.debug("HarborSolver: timed-out agent flush signal failed", exc_info=True)
            if not agent_task.done():
                agent_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await agent_task

        agent_error: Exception | None = None
        if agent_task.done() and not agent_task.cancelled():
            exc = agent_task.exception()
            if exc is not None:
                logger.warning("HarborSolver: agent.run() raised %s: %s", type(exc).__name__, exc)
                agent_error = exc

        return timed_out, agent_error

    async def _inject_skill(self, sandbox: "Sandbox", prompt: str) -> str:
        """Upload skill files to the container and prepend the skill trigger.

        If ``skill_dir`` is configured, files are uploaded to
        ``/skills/<skill>/`` before the agent starts so it can read them.
        The instruction receives a preamble pointing at the skill file.

        After uploading, verifies the primary SKILL.md landed in the container
        and logs a clear ERROR if it didn't so failures aren't silent.
        """
        if not self._skill:
            return prompt

        import shlex

        skill_dest = f"/skills/{self._skill}"
        await sandbox.exec(f"mkdir -p {shlex.quote(skill_dest)}", timeout_sec=10)

        if self._skill_dir and self._skill_dir.is_dir():
            uploaded: list[str] = []
            for skill_file in sorted(self._skill_dir.rglob("*")):
                if skill_file.is_file():
                    rel = skill_file.relative_to(self._skill_dir)
                    dest_path = f"{skill_dest}/{rel}"
                    if str(rel.parent) != ".":
                        parent_dest = f"{skill_dest}/{rel.parent}"
                        await sandbox.exec(f"mkdir -p {shlex.quote(parent_dest)}", timeout_sec=10)
                    await sandbox.upload(skill_file, dest_path)
                    uploaded.append(str(rel))
            logger.info(
                "HarborSolver: uploaded skill '%s' (%d files) from %s → %s",
                self._skill,
                len(uploaded),
                self._skill_dir,
                skill_dest,
            )
            # Verify the primary SKILL.md actually landed
            verify = await sandbox.exec(
                f"ls -la {skill_dest}/SKILL.md 2>&1 && cat {skill_dest}/SKILL.md | head -3",
                timeout_sec=10,
            )
            if verify.return_code == 0:
                logger.info(
                    "HarborSolver: skill injection verified — %s/SKILL.md present: %s",
                    skill_dest,
                    verify.stdout.strip()[:120] if verify.stdout else "",
                )
            else:
                logger.error(
                    "HarborSolver: skill injection FAILED — %s/SKILL.md not found after upload "
                    "(rc=%d stdout=%s). Check sandbox.upload() for this sandbox type.",
                    skill_dest,
                    verify.return_code,
                    (verify.stdout or verify.stderr or "")[:200],
                )
        elif self._skill_dir:
            logger.warning(
                "HarborSolver: skill_dir '%s' does not exist — skipping upload",
                self._skill_dir,
            )

        trigger = (
            f"Before working on this task, read the skill guidance at "
            f"`{skill_dest}/SKILL.md` and apply it throughout your work.\n\n"
        )
        return trigger + prompt

    async def solve(
        self,
        task: SeedResult,
        sandbox: Sandbox | None = None,
    ) -> SolveResult:
        if sandbox is None:
            raise RuntimeError("HarborSolver requires a sandbox.")

        from harbor.models.agent.context import AgentContext

        from nemo_evaluator.solvers.harbor_adapter import SandboxEnvironmentAdapter

        _silence_harbor_debug()

        t0 = time.monotonic()
        logs_dir = Path(tempfile.mkdtemp(prefix="eval_harbor_"))
        agent_logs_dir = logs_dir / "agent"
        agent_logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            resolved_url = sandbox.resolved_endpoint_url("MODEL_BASE_URL") or (
                sandbox.resolve_outside_endpoint(self._model_url) if self._model_url else self._model_url
            )

            override: dict[str, str] = {}
            if resolved_url:
                override["LLM_BASE_URL"] = resolved_url

            adapter = SandboxEnvironmentAdapter(
                sandbox,
                session_id=task.metadata["task_id"],
                logs_dir=logs_dir,
                default_timeout=self._timeout,
                persistent_env=self._container_env,
                override_env=override,
            )

            agent = self._create_agent(agent_logs_dir, model_url=resolved_url)
            await sandbox.exec(
                "mkdir -p /logs/agent /logs/verifier /logs/artifacts",
                timeout_sec=10,
            )

            # Ensure python3 >= 3.12 for openhands-sdk and install stdbuf
            # (coreutils).  pyenv shims are handled by overwriting the
            # current `python3` location directly.
            if self._harbor_agent.lower() == "openhands-sdk":
                hack_result = await sandbox.exec(
                    "if python3 -c 'import sys; exit(0 if sys.version_info >= (3,12) else 1)' 2>/dev/null; then "
                    "  echo 'System python3 is >=3.12, no shim needed'; "
                    "else "
                    "  OLD_PY3=$(which python3 2>/dev/null || echo '') && "
                    "  (command -v curl >/dev/null 2>&1 || "
                    "    (apt-get update -qq && apt-get install -y -qq curl) 2>/dev/null || "
                    "    apk add --no-cache curl 2>/dev/null || "
                    "    yum install -y curl 2>/dev/null || "
                    "    dnf install -y curl 2>/dev/null || "
                    "    true) && "
                    "  (curl -LsSf https://astral.sh/uv/install.sh || wget -qO- https://astral.sh/uv/install.sh) | sh && "
                    '  export PATH="$HOME/.local/bin:$PATH" && '
                    "  uv python install 3.13 && "
                    "  mkdir -p /usr/local/bin && "
                    '  UV_PY=$(uv python find 3.13) && [ -n "$UV_PY" ] && '
                    '  ln -sf "$UV_PY" /usr/local/bin/python3 && '
                    '  if [ -n "$OLD_PY3" ] && [ "$OLD_PY3" != "/usr/local/bin/python3" ]; then '
                    '    ln -sf "$UV_PY" "$OLD_PY3"; '
                    "  fi && "
                    "  hash -r && "
                    "  echo 'Shimmed python3 -> Python 3.13'; "
                    "fi && "
                    "command -v stdbuf >/dev/null 2>&1 || ("
                    "  (apt-get update -qq && apt-get install -y -qq coreutils) 2>/dev/null || "
                    "  apk add --no-cache coreutils 2>/dev/null || "
                    "  yum install -y coreutils 2>/dev/null || "
                    "  dnf install -y coreutils 2>/dev/null || true"
                    ") || true",
                    timeout_sec=300,
                )
                logger.info(
                    "openhands-sdk HACK (rc=%d): %s%s",
                    hack_result.return_code,
                    hack_result.stdout[:500] if hack_result.stdout else "",
                    f" | stderr: {hack_result.stderr[:500]}" if hack_result.stderr else "",
                )
                ver_result = await sandbox.exec("python3 --version 2>&1 && which python3", timeout_sec=10)
                logger.info("python3 after shim: %s", ver_result.stdout.strip() if ver_result.stdout else "N/A")

            await agent.setup(adapter)

            if self._harbor_agent.lower() == "openhands-sdk":
                await _patch_openhands_sdk(sandbox, cmd_timeout=self._cmd_timeout)

            context = AgentContext()

            agent_error: Exception | None = None
            agent_timed_out = False
            task_timeout = task.metadata.get("agent_timeout_sec")
            if task_timeout is not None and not isinstance(task_timeout, (int, float)):
                logger.warning("agent_timeout_sec in metadata is not numeric: %r, ignoring", task_timeout)
                task_timeout = None
            run_timeout = _resolve_agent_timeout(
                self._timeout_strategy,
                self._run_timeout,
                task_timeout,
                self._max_agent_timeout,
            )
            logger.info(
                "HarborSolver: timeout resolved: strategy=%s nel=%.0fs task=%s cap=%s → effective=%.0fs",
                self._timeout_strategy,
                self._run_timeout,
                f"{task_timeout:.0f}s" if task_timeout is not None else "n/a",
                f"{self._max_agent_timeout:.0f}s" if self._max_agent_timeout is not None else "n/a",
                run_timeout,
            )
            jitter = random.uniform(0, min(120.0, run_timeout * 0.02))
            effective_timeout = run_timeout + jitter

            prompt = await self._inject_skill(sandbox, task.prompt)
            agent_task = asyncio.create_task(agent.run(prompt, adapter, context))
            agent_t0 = time.monotonic()
            agent_timed_out, agent_error = await self._wait_for_agent(
                agent_task,
                sandbox,
                agent_t0,
                effective_timeout,
                jitter,
            )

            # Capture git diff before downloading logs (agent may have
            # modified /testbed in SWE-bench and similar tasks).
            workspace_diff = ""
            if sandbox.is_running:
                try:
                    workspace_diff = await _capture_workspace_diff(sandbox)
                except Exception:
                    logger.debug("workspace diff capture failed", exc_info=True)

            # Download container-side logs (no-op when host-mounted)
            if not adapter.is_mounted and sandbox.is_running:
                await _download_agent_logs(sandbox, agent_logs_dir)

            # Let agent parse its own logs into context
            if context.is_empty() and hasattr(agent, "populate_context_post_run"):
                try:
                    agent.populate_context_post_run(context)
                except Exception:
                    logger.warning("populate_context_post_run failed", exc_info=True)

            # Recover trajectory / tokens / response from files (single pass)
            recovered = _recover_from_logs(agent_logs_dir)

            # Prefer file-based trajectory (already ATIF) over
            # context.rollout_details (raw token IDs for SFT).
            trajectory = recovered["trajectory"]
            if not trajectory and context.rollout_details:
                raw_details = [dict(d) if isinstance(d, dict) else d for d in context.rollout_details]
                trajectory = build_atif_trajectory(
                    raw_details,
                    agent_name=self._harbor_agent,
                    prompt_tokens=context.n_input_tokens or 0,
                    completion_tokens=context.n_output_tokens or 0,
                )
            if not trajectory and context.metadata:
                trajectory = build_atif_trajectory(
                    [{"source": "agent", "message": str(context.metadata)}],
                    agent_name=self._harbor_agent,
                )

            # Store workspace diff in trajectory metadata (observability only).
            if trajectory and workspace_diff:
                doc = trajectory[0] if isinstance(trajectory, list) and trajectory else None
                if isinstance(doc, dict):
                    fm = doc.setdefault("final_metrics", {})
                    fm["workspace_diff_preview"] = workspace_diff[:100_000]

            # OpenHands: flag zero-token agent turns — symptom of a context
            # reset/condensation that failed to fire (litellm vs API token
            # mismatch). Observability only; not classified as infra.
            if self._harbor_agent.lower() in ("openhands", "openhands-sdk") and trajectory:
                zero_token_turns = _count_zero_token_agent_turns(trajectory)
                if zero_token_turns:
                    logger.warning(
                        "HarborSolver: OpenHands produced %d zero-token agent turn(s) — "
                        "possible context-reset/condensation failure (litellm vs API token "
                        "mismatch). See trajectory final_metrics.zero_token_turns.",
                        zero_token_turns,
                    )
                    doc = trajectory[0] if isinstance(trajectory, list) and trajectory else None
                    if isinstance(doc, dict):
                        doc.setdefault("final_metrics", {})["zero_token_turns"] = zero_token_turns

            # Response: prefer actual agent text, sentinel if empty/echo.
            response = _extract_response(context) or recovered["response"]
            if not response or _is_prompt_echo(response, task.prompt):
                response = "[workspace modified]" if workspace_diff else ""

            # Tokens: context first, file fallback
            prompt_tokens = context.n_input_tokens or recovered["prompt_tokens"]
            completion_tokens = context.n_output_tokens or recovered["completion_tokens"]

            latency_ms = (time.monotonic() - t0) * 1000

            # Timeout with zero progress → model is likely dead.
            if agent_timed_out and not workspace_diff and prompt_tokens + completion_tokens == 0:
                raise InfraError(
                    f"Agent made no progress before run_timeout ({self._run_timeout:.0f}s). Model may be unreachable."
                )

            # Detect partial progress with zero final tokens — vLLM may
            # have died mid-solve (workspace changed from earlier turns but
            # the last inference produced nothing).
            _confirmed_zero_tokens = (context.n_output_tokens is not None and context.n_output_tokens == 0) or (
                context.n_output_tokens is None and recovered["completion_tokens"] == 0 and not recovered["response"]
            )

            error = None
            error_kind = ErrorKind.NONE
            if agent_error is not None:
                etype = type(agent_error).__name__
                if etype in _INFRA_ERROR_NAMES:
                    raise InfraError(f"Agent infrastructure failure: {etype}: {agent_error}") from agent_error
                error = f"Agent crashed: {etype}: {agent_error}"
                logger.warning("HarborSolver: %s", error)
            elif crash_error := _error_from_crash_marker(agent_logs_dir):
                error = crash_error
                logger.warning("HarborSolver: %s", error)
            elif agent_timed_out and workspace_diff and _confirmed_zero_tokens:
                error = (
                    f"Agent timed out with workspace changes but 0 completion "
                    f"tokens (run_timeout={self._run_timeout:.0f}s). "
                    f"Model may have died mid-solve."
                )
                error_kind = ErrorKind.INFRA
                logger.warning("HarborSolver: %s", error)
            elif agent_timed_out and workspace_diff:
                logger.info(
                    "HarborSolver: agent timed out after %.0fs but produced "
                    "workspace changes — submitting for verification",
                    self._run_timeout,
                )
                if not trajectory:
                    logger.warning(
                        "HarborSolver: timeout trajectory flush produced no events "
                        "(empty trajectory); verifying on workspace patch only"
                    )
                error_kind = ErrorKind.SOLVE_TIMEOUT
            elif not response and prompt_tokens + completion_tokens == 0:
                error = "Agent produced no output (0 tokens, empty response). Check agent logs for details."
                logger.warning("HarborSolver: %s", error)
            elif agent_timed_out:
                logger.info(
                    "HarborSolver: agent timed out after %.0fs without workspace changes "
                    "(prompt_tokens=%d completion_tokens=%d) — submitting for verification",
                    self._run_timeout,
                    prompt_tokens,
                    completion_tokens,
                )
                if not trajectory:
                    logger.warning(
                        "HarborSolver: timeout trajectory flush produced no events "
                        "(empty trajectory); verifying on workspace patch only"
                    )
                error_kind = ErrorKind.SOLVE_TIMEOUT

            return SolveResult(
                response=response,
                model_response=ModelResponse(
                    content=response,
                    model=self._model_id,
                    total_tokens=prompt_tokens + completion_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=round(latency_ms, 2),
                ),
                trajectory=trajectory,
                error=error,
                error_kind=error_kind,
            )

        except InfraError as exc:
            logger.warning("HarborSolver: infra failure: %s", exc)
            latency_ms = (time.monotonic() - t0) * 1000

            workspace_diff = ""
            try:
                if sandbox.is_running:
                    workspace_diff = await _capture_workspace_diff(sandbox)
                    await _download_agent_logs(sandbox, agent_logs_dir)
            except Exception:
                logger.debug("Post-failure recovery failed", exc_info=True)

            recovered = _recover_from_logs(agent_logs_dir)
            trajectory = recovered["trajectory"] or build_atif_trajectory(
                steps=[{"source": "system", "message": str(exc)}],
                agent_name=self._harbor_agent,
                status="error",
            )

            if trajectory and workspace_diff:
                doc = trajectory[0] if isinstance(trajectory, list) and trajectory else None
                if isinstance(doc, dict):
                    fm = doc.setdefault("final_metrics", {})
                    fm["workspace_diff_preview"] = workspace_diff[:100_000]

            response = recovered["response"]
            if not response or _is_prompt_echo(response, ""):
                response = "[workspace modified]" if workspace_diff else ""

            return SolveResult(
                response=response,
                model_response=ModelResponse(
                    content=response,
                    model=self._model_id,
                    total_tokens=recovered["prompt_tokens"] + recovered["completion_tokens"],
                    completion_tokens=recovered["completion_tokens"],
                    latency_ms=round(latency_ms, 2),
                ),
                trajectory=trajectory,
                error=str(exc),
                error_kind=ErrorKind.INFRA,
            )

        except GracefulError as exc:
            logger.warning("HarborSolver: graceful failure: %s", exc)
            latency_ms = (time.monotonic() - t0) * 1000

            workspace_diff = ""
            try:
                if sandbox.is_running:
                    workspace_diff = await _capture_workspace_diff(sandbox)
                    await _download_agent_logs(sandbox, agent_logs_dir)
            except Exception:
                logger.debug("Post-failure recovery failed", exc_info=True)

            recovered = _recover_from_logs(agent_logs_dir)
            trajectory = recovered["trajectory"] or build_atif_trajectory(
                steps=[{"source": "system", "message": str(exc)}],
                agent_name=self._harbor_agent,
                status="error",
            )

            if trajectory and workspace_diff:
                doc = trajectory[0] if isinstance(trajectory, list) and trajectory else None
                if isinstance(doc, dict):
                    fm = doc.setdefault("final_metrics", {})
                    fm["workspace_diff_preview"] = workspace_diff[:100_000]

            response = recovered["response"]
            if not response or _is_prompt_echo(response, ""):
                response = "[workspace modified]" if workspace_diff else ""

            return SolveResult(
                response=response,
                model_response=ModelResponse(
                    content=response,
                    model=self._model_id,
                    total_tokens=recovered["prompt_tokens"] + recovered["completion_tokens"],
                    completion_tokens=recovered["completion_tokens"],
                    latency_ms=round(latency_ms, 2),
                ),
                trajectory=trajectory,
                error=str(exc),
            )

        except Exception:
            logger.exception("HarborSolver.solve() system error — will retry")
            raise

    async def close(self) -> None:
        pass
