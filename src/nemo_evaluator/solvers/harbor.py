"""HarborSolver: runs Harbor-compatible agents inside a nel Sandbox."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.errors import GracefulError
from nemo_evaluator.observability.types import ModelResponse
from nemo_evaluator.solvers.base import SolveResult
from nemo_evaluator.solvers.trajectory_util import build_atif_trajectory

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import SeedResult
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


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
    """Return *explicit* if non-empty, else probe common env vars."""
    if explicit:
        return explicit
    for var in ("LLM_API_KEY", "NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        val = os.environ.get(var)
        if val:
            return val
    return None


def _ensure_env(api_key: str | None, model_url: str | None, model_id: str | None) -> None:
    """Set ``LLM_*`` env vars that Harbor agents expect.

    Called once — these are process-wide settings that must stay set for the
    lifetime of concurrent solve() calls. A save/restore context manager is
    unsafe here because ``os.environ`` is process-global and concurrent
    tasks would clobber each other's cleanup.

    Harbor uses litellm as its LLM layer, so ``LLM_MODEL`` gets the
    ``openai/`` prefix when a custom ``model_url`` is set (litellm needs
    the provider hint for OpenAI-compatible endpoints).
    """
    key = _resolve_api_key(api_key)
    if not key:
        key = "no-key-needed"
        logger.info("No API key found — using dummy key for self-hosted model")
    os.environ["LLM_API_KEY"] = key
    if model_url:
        os.environ["LLM_BASE_URL"] = model_url
    if model_id:
        mid = model_id
        if model_url and not mid.startswith("openai/"):
            mid = f"openai/{mid}"
        os.environ["LLM_MODEL"] = mid
    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ.setdefault("LITELLM_TELEMETRY", "false")
    os.environ.setdefault("SECURITY_CONFIRMATION_MODE", "false")
    os.environ.setdefault("SECURITY_ENABLE_SECURITY_ANALYZER", "false")


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

    remote_tar = "/tmp/_nel_agent_logs.tar.gz"
    rc = await sandbox.exec(
        f"tar czf {remote_tar} -C /logs/agent .",
        timeout_sec=120,
    )
    if rc.return_code != 0:
        logger.error("tar failed (rc=%d): %s", rc.return_code, rc.stderr or rc.stdout)
        return

    import tarfile

    local_tar = Path(tempfile.mktemp(suffix=".tar.gz"))
    try:
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                await sandbox.download(remote_tar, local_tar)
                dest.mkdir(parents=True, exist_ok=True)
                with tarfile.open(local_tar, "r:gz") as tar:
                    tar.extractall(dest)
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


# ---------------------------------------------------------------------------
# Trajectory / token / response recovery  (agent-agnostic)
# ---------------------------------------------------------------------------
#
# Each Harbor agent writes its own logs and converts them to ATIF via
# ``populate_context_post_run()``.  NEL's job here is simple:
#   1. Read the ATIF trajectory.json the agent produced.
#   2. If nothing structured exists, grab the largest .txt as an error log.
#
# Agent-specific parsing (OpenHands completions/, sessions/events/, etc.)
# is deliberately NOT done here — that is the agent's responsibility.
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
    logging.getLogger("harbor").setLevel(logging.INFO)


class HarborSolver:
    """Runs any Harbor agent inside a nel Sandbox.

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
    ) -> None:
        _check_harbor_installed()
        self._harbor_agent = harbor_agent
        self._harbor_agent_kwargs = harbor_agent_kwargs or {}
        self._model_url = model_url
        self._model_id = model_id
        self._timeout = timeout
        self._run_timeout = run_timeout or timeout
        self._container_env = dict(container_env or {})
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
        _ensure_env(self._api_key, self._model_url, self._model_id)

    def _create_agent(self, logs_dir: Path, *, model_url: str = "") -> Any:
        from harbor.agents.factory import AgentFactory

        kwargs = dict(self._harbor_agent_kwargs)
        model_id = self._model_id
        url = model_url or self._model_url
        if url and "model_name" not in kwargs and model_id and not model_id.startswith("openai/"):
            model_id = f"openai/{model_id}"
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
                "max_input_tokens": self._max_input_tokens or 131072,
                "max_output_tokens": self._max_output_tokens or 16384,
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

    async def solve(
        self,
        task: SeedResult,
        sandbox: Sandbox | None = None,
    ) -> SolveResult:
        if sandbox is None:
            raise RuntimeError("HarborSolver requires a sandbox.")

        from harbor.models.agent.context import AgentContext
        from nemo_evaluator.solvers.harbor_adapter import SandboxEnvironmentAdapter

        t0 = time.monotonic()
        logs_dir = Path(tempfile.mkdtemp(prefix="nel_harbor_"))
        agent_logs_dir = logs_dir / "agent"
        agent_logs_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Rewrite the proxy/model URL for the sandbox's network topology
            # (e.g. 127.0.0.1 → host.docker.internal for Docker bridge).
            resolved_url = sandbox.resolve_outside_endpoint(self._model_url) if self._model_url else self._model_url
            # Harbor agents read LLM_BASE_URL from os.environ directly
            # (not from kwargs), so we must update the process env.
            if resolved_url:
                os.environ["LLM_BASE_URL"] = resolved_url

            adapter = SandboxEnvironmentAdapter(
                sandbox,
                logs_dir=logs_dir,
                default_timeout=self._timeout,
                persistent_env=self._container_env,
            )

            agent = self._create_agent(agent_logs_dir, model_url=resolved_url)
            await sandbox.exec(
                "mkdir -p /logs/agent /logs/verifier /logs/artifacts",
                timeout_sec=10,
            )

            # HACK: Pre-install openhands-sdk with uv + Python 3.13 so the
            # harbor install script's "already installed" check passes and it
            # skips its own venv creation (which uses the system python3 that
            # may be too old for openhands-sdk's >3.12 requirement).
            # Remove once the harbor install script is updated to use uv.
            if self._harbor_agent.lower() == "openhands-sdk":
                await sandbox.exec(
                    "if [ -f /opt/openhands-sdk-venv/bin/python ] "
                    "&& /opt/openhands-sdk-venv/bin/python -c 'import openhands.sdk' 2>/dev/null; then "
                    "  echo 'openhands-sdk venv already present'; "
                    "else "
                    "  apt-get update -qq && apt-get install -y -qq curl && "
                    "  curl -LsSf https://astral.sh/uv/install.sh | sh && "
                    '  export PATH="$HOME/.local/bin:$PATH" && '
                    "  uv python install 3.13 && "
                    "  uv venv /opt/openhands-sdk-venv --python 3.13 && "
                    "  . /opt/openhands-sdk-venv/bin/activate && "
                    "  uv pip install openhands-sdk openhands-tools && "
                    "  echo 'openhands-sdk pre-installed with uv + Python 3.13'; "
                    "fi",
                    timeout_sec=300,
                )

            await agent.setup(adapter)
            context = AgentContext()

            agent_timed_out = False
            jitter = random.uniform(0, min(120.0, self._run_timeout * 0.02))
            effective_timeout = self._run_timeout + jitter
            try:
                await asyncio.wait_for(
                    agent.run(task.prompt, adapter, context),
                    timeout=effective_timeout,
                )
            except (asyncio.TimeoutError, TimeoutError):
                elapsed = time.monotonic() - t0
                logger.warning(
                    "HarborSolver: agent.run() timed out after %.0fs (run_timeout=%.0fs+%.0fs jitter) — collecting partial results",
                    elapsed,
                    self._run_timeout,
                    jitter,
                )
                agent_timed_out = True

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

            # Timeout with zero progress → raise so the eval loop retries
            # on a fresh sandbox (the model may have been temporarily down).
            # Timeout WITH partial work → return normally for verification.
            if agent_timed_out and not workspace_diff and prompt_tokens + completion_tokens == 0:
                raise RuntimeError(
                    f"Agent made no progress before run_timeout ({self._run_timeout:.0f}s). Model may be unreachable."
                )

            error = None
            if agent_timed_out and workspace_diff:
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
