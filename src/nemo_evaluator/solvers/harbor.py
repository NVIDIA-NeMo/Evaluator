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
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.errors import GracefulError, InfraError
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import ErrorKind, SolveResult
from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import SeedResult
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


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


async def _patch_openhands_sdk(sandbox: "Sandbox", *, cmd_timeout: float | None = None) -> None:
    """Apply runtime patches to the OpenHands SDK inside the sandbox.

    1. **Prevent premature FINISHED on text-only responses** — when the
       LLM returns text without a tool call the SDK sets
       ``execution_status = FINISHED`` and stops.  Reasoning models often
       produce intermediate text before their next tool call, so the
       patched loop continues until ``finish`` is called or
       ``max_iterations`` is reached.

    2. **Capture reasoning in ATIF trajectory** — the runner's event
       serialization drops ``reasoning_content`` / ``thinking_blocks``.
       Both the event-to-dict conversion and ``build_trajectory`` are
       patched so reasoning appears in the output JSON.

    3. **Preserve reasoning in conversation history** — some reasoning
       parsers move ``<tool_call>`` blocks inside ``<think>`` tags into
       ``reasoning_content`` instead of ``tool_calls``.  The SDK then
       drops ``reasoning_content`` on the next serialization pass,
       causing the model to lose its chain-of-thought.  The patch wraps
       ``reasoning_content`` in ``<think>`` tags and prepends it to
       ``content`` so it survives round-trips.

    4. **Enforce 300 s hard timeout on terminal commands** — the SDK
       only applies a hard timeout when the model passes an explicit
       ``timeout`` parameter.  The patch imposes a 300 s ceiling on
       every command to prevent a single long-running process from
       consuming the entire ``run_timeout`` budget.

    5. **Disable default visualizer + stuck detection** — the SDK's
       ``DefaultConversationVisualizer`` renders every event through
       ``rich``, whose grapheme splitter is pathologically slow on long
       text containing U+200B adjacent to URLs.  Observed on 26 django
       SWE-bench prompts: the first ``send_message()`` burns 100 % CPU
       inside ``rich.cells.split_graphemes`` for the full 90 min
       ``run_timeout`` without ever reaching the LLM.  ``stuck_detection``
       is also disabled in the same patch because its heuristic
       mis-flags reasoning-model turns.
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
        "old = 'conversation = Conversation(**conv_kwargs)'\n"
        'new = \'conv_kwargs["stuck_detection"] = False\\n'
        '    conv_kwargs["visualizer"] = None\\n'
        "    conversation = Conversation(**conv_kwargs)'\n"
        "if old in c:\n"
        "    open(p, 'w').write(c.replace(old, new, 1))\n"
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

    # -- Patch 2: capture reasoning + per-step metrics in ATIF trajectory ----
    # The runner converts SDK events → intermediate dicts → ATIF steps but
    # never copies reasoning_content / thinking_blocks or per-turn token usage.
    # We add both at the event-conversion and step-building stages.
    #
    # Event-collection (A, B): extract reasoning_content and LLM usage from
    # the SDK event object and store in the intermediate dict.
    # build_trajectory (C): propagate both to ATIF step fields.

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
    '            _u = event.get("usage") or {}\\n'
    '            _pt = int(_u.get("prompt_tokens", 0) or 0) if isinstance(_u, dict) else 0\\n'
    '            _ct = int(_u.get("completion_tokens", 0) or 0) if isinstance(_u, dict) else 0\\n'
    '            if _pt or _ct:\\n'
    '                step["metrics"] = {"prompt_tokens": _pt, "completion_tokens": _ct}\\n'
    '            steps.append(step)\\n'
    '            step_id += 1\\n'
    '\\n'
    '        elif event_type == "tool_result":'
)

ok_a = old_a in c
c = c.replace(old_a, new_a, 1) if ok_a else c
ok_b = old_b in c
c = c.replace(old_b, new_b, 1) if ok_b else c
ok_c = old_c in c
c = c.replace(old_c, new_c, 1) if ok_c else c
open(p, 'w').write(c)
print(f'reasoning+metrics: msg={ok_a} action={ok_b} traj={ok_c}')
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

    # -- Patch 3: preserve reasoning_content in conversation history -------
    # When send_reasoning_content is False (nemotron is not in the list),
    # the SDK silently drops reasoning_content from assistant messages.
    # Patch to_chat_dict() to wrap reasoning_content in <think> tags and
    # prepend to content so it survives LiteLLM serialization round-trips.

    _reasoning_in_content_script = """\
import glob, sys
fs = glob.glob('/opt/openhands-sdk-venv/lib/python*/site-packages/openhands/sdk/llm/message.py')
p = fs[0] if fs else ''
assert p, 'message.py not found'
c = open(p).read()

old = (
    '        # Required for model like kimi-k2-thinking\\n'
    '        if send_reasoning_content and self.reasoning_content:\\n'
    '            message_dict["reasoning_content"] = self.reasoning_content\\n'
    '\\n'
    '        return message_dict'
)

new = (
    '        # Required for model like kimi-k2-thinking\\n'
    '        if send_reasoning_content and self.reasoning_content:\\n'
    '            message_dict["reasoning_content"] = self.reasoning_content\\n'
    '\\n'
    '        # [NEL] Wrap reasoning_content in <think> tags in content so the\\n'
    '        # model sees its previous chain-of-thought on retry turns.\\n'
    '        if not send_reasoning_content and self.role == "assistant" and self.reasoning_content:\\n'
    '            _wrapped = f"<think>{self.reasoning_content}</think>"\\n'
    '            _c = message_dict.get("content")\\n'
    '            if isinstance(_c, list):\\n'
    '                _c.insert(0, {"type": "text", "text": _wrapped})\\n'
    '            elif isinstance(_c, str):\\n'
    '                message_dict["content"] = _wrapped + _c\\n'
    '            else:\\n'
    '                message_dict["content"] = _wrapped\\n'
    '\\n'
    '        return message_dict'
)

ok = old in c
if ok:
    c = c.replace(old, new, 1)
    open(p, 'w').write(c)
print(f'reasoning_in_content={ok} at {p}')
"""
    encoded4 = base64.b64encode(_reasoning_in_content_script.encode()).decode()
    r4 = await sandbox.exec(
        f"echo {encoded4} | base64 -d | python3",
        timeout_sec=10,
    )
    stdout4 = (r4.stdout or "").strip()
    logger.info("Reasoning-in-content patch: %s", stdout4)
    if r4.return_code != 0 or "False" in stdout4:
        logger.warning(
            "Reasoning-in-content patch problem (rc=%d): %s",
            r4.return_code,
            stdout4 or (r4.stderr or "")[:300],
        )

    # -- Patch 4: hard timeout ceiling on terminal commands ------------------
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
            logger.info("Trajectory loaded from %s", tf.name)
            out["trajectory"] = [parsed["doc"]]
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
        cmd_timeout: float | None = None,
        timeout_strategy: str = "override",
        max_agent_timeout: float | None = None,
    ) -> None:
        _check_harbor_installed()
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
        self._max_input_tokens = max_input_tokens
        self._max_output_tokens = max_output_tokens
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

        kwargs = dict(self._harbor_agent_kwargs)
        url = model_url or self._model_url
        model_id = _model_id_for_openai(self._model_id, bool(url), agent=self._harbor_agent) if self._model_id else ""
        if "model_name" not in kwargs and model_id:
            kwargs["model_name"] = model_id
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
        t0: float,
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
            remaining = effective_timeout - (time.monotonic() - t0)
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
                time.monotonic() - t0,
                effective_timeout - jitter,
                jitter,
                self._timeout_strategy,
            )
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

            agent_task = asyncio.create_task(agent.run(task.prompt, adapter, context))
            agent_timed_out, agent_error = await self._wait_for_agent(
                agent_task,
                sandbox,
                t0,
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

            _infra_error_names = {
                "ServiceUnavailableError",
                "ConnectionError",
                "TimeoutError",
                "ConnectError",
                "ReadTimeout",
                "APIConnectionError",
            }

            error = None
            error_kind = ErrorKind.NONE
            if agent_error is not None:
                etype = type(agent_error).__name__
                if etype in _infra_error_names:
                    raise InfraError(f"Agent infrastructure failure: {etype}: {agent_error}") from agent_error
                error = f"Agent crashed: {etype}: {agent_error}"
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
            elif not response and prompt_tokens + completion_tokens == 0:
                error = "Agent produced no output (0 tokens, empty response). Check agent logs for details."
                logger.warning("HarborSolver: %s", error)

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
