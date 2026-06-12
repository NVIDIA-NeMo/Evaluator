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
"""ContainerEnvironment: run legacy eval-factory containers as opaque environments.

Wraps any legacy evaluation container image, runs it via ``docker run``,
and injects a **legacy-format** ``run_config.yaml`` so the container's
``nemo-evaluator run_eval`` entrypoint can consume it directly.

The generated run-config follows the canonical Evaluator schema::

    config:
      type: <task>
      output_dir: /results
      params: { ... }          # optional pass-through
    target:
      api_endpoint:
        url: <model_url>
        model_id: <model_id>
        api_key_name: NEMO_API_KEY
        type: chat              # or completions

Usage in NEL config:
    benchmarks:
      - name: "container://registry/image:tag#eval_type"
        solver:
          type: container
          service: nemotron
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

import yaml

from nemo_evaluator.completions_guard import normalize_adapter_config_for_endpoint
from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

logger = logging.getLogger(__name__)

_CONTAINER_RESULTS_DIR = "/results"
_CONTAINER_CONFIG_PATH = "/config/run_config.yaml"
_API_KEY_ENV = "NEMO_API_KEY"

_NEL_PROTOCOL_TO_LEGACY_TYPE = {
    "chat_completions": "chat",
    "completions": "completions",
    "responses": "chat",
}

# Matches `-e KEY=VALUE` argv pairs (either joined or split) so values don't leak to logs.
_REDACT_ENV_VALUE = re.compile(r"(-e\s+\w+=)\S+")


def build_legacy_run_config(
    task: str,
    model_url: str,
    model_id: str,
    endpoint_type: str,
    extra_params: dict[str, Any] | None = None,
    adapter_config: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build a legacy Evaluator ``run_config.yaml`` dict.

    Module-level so callers outside ``ContainerEnvironment`` (slurm_gen
    pre-rendering at submit time) can produce the same yaml the
    in-process docker dispatch path produces.

    The container's ``nemo-evaluator run_eval --run_config <path>``
    entrypoint consumes this and merges it with the baked-in
    ``framework.yml`` defaults.

    ``api_key_name`` is only emitted when *api_key* is truthy.  Mirrors
    v1 launcher's behaviour for self-deployed vLLM services where the
    hydra config sets ``target.api_endpoint.api_key_name: null`` — the
    harness then skips the ``os.environ[NEMO_API_KEY]`` lookup, which
    would otherwise raise ``ValueError`` when no real key exists.
    """
    api_endpoint: dict[str, Any] = {
        "url": model_url,
        "model_id": model_id,
        "type": endpoint_type,
    }
    if api_key:
        api_endpoint["api_key_name"] = _API_KEY_ENV
    # Chat-template controls are inert on a text-completions endpoint and would
    # silently leave reasoning enabled; strip (or reject under strict mode)
    # before the container's in-process adapter ever sees them.
    adapter_config = normalize_adapter_config_for_endpoint(adapter_config, endpoint_type)
    if adapter_config:
        api_endpoint["adapter_config"] = adapter_config
    run_config: dict[str, Any] = {
        "config": {
            "type": task,
            "output_dir": _CONTAINER_RESULTS_DIR,
        },
        "target": {"api_endpoint": api_endpoint},
    }
    if extra_params:
        run_config["config"]["params"] = extra_params
    return run_config


def parse_legacy_results(results_dir: Path) -> dict[str, Any]:
    """Parse ``results.yml`` / ``results.json`` from a legacy harness.

    Returns ``{"scores": {...}, "samples": <int>, "repeats": <int>}``.
    Module-level so post-job report steps can call it without
    instantiating ``ContainerEnvironment``.
    """
    results_file = results_dir / "results.yml"
    results_json = results_dir / "results.json"

    raw: dict[str, Any] = {}
    if results_file.exists():
        raw = yaml.safe_load(results_file.read_text()) or {}
    elif results_json.exists():
        raw = json.loads(results_json.read_text())

    scores = ContainerEnvironment._extract_scores(raw)
    samples = ContainerEnvironment._extract_sample_count(raw, scores)
    repeats = _extract_repeats(raw)
    return {"scores": scores, "samples": samples, "repeats": repeats}


def _extract_repeats(raw: dict[str, Any]) -> int:
    """Read ``config.params.extra.num_repeats`` from a legacy harness
    output.  The eval-factory wrapper round-trips the run_config back
    into ``results.yml`` so this field is the source of truth for how
    many times each problem was actually evaluated.  Defaults to 1
    matching :attr:`BenchmarkConfig.repeats` for native-env bundles.
    """
    cfg = raw.get("config") or {}
    params = cfg.get("params") or {}
    extra = params.get("extra") or {}
    n = extra.get("num_repeats")
    try:
        return int(n) if n else 1
    except (TypeError, ValueError):
        return 1


def build_legacy_bundle(image: str, task: str, results_dir: Path) -> dict[str, Any]:
    """Build a NEL result bundle from a legacy harness ``results.yml``.

    The bundle structure matches what
    :meth:`ContainerEnvironment.run_batch` returns for the in-process
    dispatch path, so post-job conversion produces artifacts identical
    to the live-orchestrator path.
    """
    parsed = parse_legacy_results(results_dir)
    rows = _parse_legacy_output_rows(results_dir)
    name = f"container/{task or image.split('/')[-1].split(':')[0]}"
    bundle: dict[str, Any] = {
        "benchmark": {
            "name": name,
            "samples": parsed["samples"],
            "repeats": parsed["repeats"],
            "scores": parsed["scores"],
        },
        "config": {
            "benchmark": name,
            "image": image,
            "task": task,
            "framework": "container",
        },
    }
    if rows:
        bundle["_results"] = rows
        bundle["n_results"] = len(rows)
    return bundle


def _parse_legacy_output_rows(results_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("eval-results/**/output*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed legacy output row in %s: %s", path, exc)
                    continue
                if isinstance(record, dict):
                    rows.append(record)

    repeat_counts: dict[Any, int] = {}
    normalized: list[dict[str, Any]] = []
    for fallback_idx, row in enumerate(rows):
        result = dict(row)
        problem_idx = result.get("problem_idx", fallback_idx)
        result["problem_idx"] = problem_idx
        if result.get("repeat") is None:
            result["repeat"] = repeat_counts.get(problem_idx, 0)
        repeat_counts[problem_idx] = repeat_counts.get(problem_idx, 0) + 1

        reward = _legacy_row_reward(result)
        if reward is not None:
            result.setdefault("reward", reward)

        result.setdefault("prompt", result.get("problem") or result.get("question") or result.get("input", ""))
        result.setdefault(
            "model_response", result.get("generation") or result.get("response") or result.get("output", "")
        )
        if "expected_answer" not in result and "answer" in result:
            result["expected_answer"] = result["answer"]
        if "extracted_answer" not in result and "predicted_answer" in result:
            result["extracted_answer"] = result["predicted_answer"]
        result.setdefault("metadata", {})
        result.setdefault("scoring_details", {})
        normalized.append(result)

    return normalized


def _legacy_row_reward(row: dict[str, Any]) -> float | None:
    for key in ("reward", "score"):
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)

    for key in ("symbolic_correct", "correct", "is_correct", "passed", "success"):
        value = row.get(key)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, (int, float)):
            return float(value)

    return None


def _redact_cmd_for_log(cmd: list[str]) -> str:
    """Render an argv for log output with ``-e KEY=VALUE`` values redacted."""
    joined = " ".join(cmd)
    return _REDACT_ENV_VALUE.sub(r"\1<REDACTED>", joined)


async def _drain_stream(
    stream: asyncio.StreamReader | None,
    sink: list[bytes],
    log_line: Any,
) -> None:
    """Read *stream* line by line, logging each line live and buffering it.

    Legacy harnesses emit progress on stdout/stderr only at process exit
    if we use ``communicate()``; reading line by line surfaces the
    container's own logs in real time.  Lines are also accumulated into
    *sink* so :func:`_format_container_failure` can replay the tail.
    """
    if stream is None:
        return
    while True:
        try:
            line = await stream.readline()
        except (asyncio.LimitOverrunError, ValueError):
            # Line exceeds the StreamReader buffer; drain what's there and continue.
            line = await stream.read(65536)
            if not line:
                break
        if not line:
            break
        sink.append(line)
        log_line(line.decode(errors="replace").rstrip())


def _format_container_failure(stdout: bytes | None, stderr: bytes | None, tail: int = 4000) -> str:
    """Format a container's captured output for a failure log.

    Legacy harnesses emit INFO logs first and the actual traceback last,
    so we show the *tail* of each stream rather than the head — a head
    truncation cuts off exactly the error that explains the failure.
    Both streams are included because some harnesses write the traceback
    to stdout while others use stderr.
    """

    def _section(label: str, data: bytes | None) -> str:
        text = (data or b"").decode(errors="replace").strip()
        if not text:
            return f"--- container {label} (empty) ---"
        if len(text) > tail:
            text = "...[truncated]...\n" + text[-tail:]
        return f"--- container {label} (last {tail} chars) ---\n{text}"

    return f"{_section('stdout', stdout)}\n{_section('stderr', stderr)}"


def _runs_on_slurm() -> bool:
    """True when the current process is inside a SLURM allocation.

    When set, ``ContainerEnvironment`` dispatches via ``srun
    --container-image`` (Pyxis/Enroot) instead of ``docker run`` — HPC
    compute nodes (e.g., HSG) typically don't have Docker.

    To force the Docker path on a SLURM node, ``unset SLURM_JOB_ID``
    before invoking nel.
    """
    return bool(os.environ.get("SLURM_JOB_ID"))


def _resolve_pyxis_image(image: str) -> str:
    """Format an image string for Pyxis ``--container-image=``.

    File paths (``/srv/.../foo.sqsh``) pass through unchanged; registry
    URIs (``nvcr.io/x:tag``, ``registry.example.com:5005/.../x:tag``)
    get a ``docker://`` prefix so enroot imports them on first use.
    """
    if image.startswith(("/", "docker://")):
        return image
    return f"docker://{image}"


class ContainerEnvironment(EvalEnvironment):
    """Runs a legacy eval-factory container and parses its results.

    The container is launched with a mounted ``run_config.yaml`` in the
    canonical legacy Evaluator format.  Results are read from
    ``/results/results.yml`` (legacy) or ``/results/results.json``.
    """

    def __init__(
        self,
        image: str,
        task: str = "",
        model_url: str | None = None,
        model_id: str | None = None,
        api_key: str | None = None,
        endpoint_type: str = "chat",
        legacy_params: dict[str, Any] | None = None,
        adapter_config: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
        extra_mounts: list[str] | None = None,
        timeout: float = 3600.0,
    ) -> None:
        super().__init__()
        self.name = f"container/{task or image.split('/')[-1].split(':')[0]}"
        self._image = image
        self._task = task
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._endpoint_type = endpoint_type
        self._legacy_params = legacy_params or {}
        self._adapter_config = adapter_config
        self._extra_env = extra_env or {}
        self._extra_mounts = extra_mounts or []
        self._timeout = timeout

    async def dataset_size(self) -> int:
        return 0

    async def seed(self, idx: int) -> SeedResult:
        raise NotImplementedError("ContainerEnvironment uses run_batch()")

    async def verify(
        self, response: str, expected: str, sandbox: Sandbox | None = None, **metadata: Any
    ) -> VerifyResult:
        raise NotImplementedError("ContainerEnvironment uses run_batch()")

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def run_batch(self, solver: Any = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the container and parse results."""
        config = config or {}
        model_url = self._model_url or config.get("base_url", "")
        model_id = self._model_id or config.get("model", "")
        api_key = self._api_key or config.get("api_key", "")
        endpoint_type = config.get("endpoint_type", self._endpoint_type)
        extra_params = {**self._legacy_params, **config.get("params", {})}
        adapter_config = config.get("adapter_config") or self._adapter_config
        cfg_env_vars: dict[str, str] = config.get("env_vars") or {}
        cfg_mounts: dict[str, str] = config.get("mounts") or {}

        # Manual mkdtemp + try/finally + shutil.rmtree(ignore_errors=True):
        # legacy eval-factory images run as root and write artifacts
        # (cache.db, response_stats_cache/) owned by root into the bind-mounted
        # /results.  On Linux these can't be unlinked by the host's non-root
        # user.  ``TemporaryDirectory(ignore_cleanup_errors=True)`` is not
        # enough — its onexc handler still calls _resetperms which raises
        # PermissionError before the ignore_errors guard, masking the
        # successful bundle return.  ``shutil.rmtree(ignore_errors=True)`` is
        # documented "never raises" and skips the chmod entirely.
        tmpdir = tempfile.mkdtemp(prefix="nel_container_")
        try:
            results_dir = Path(tmpdir) / "results"
            results_dir.mkdir()

            run_config = self._build_legacy_run_config(
                model_url,
                model_id,
                endpoint_type,
                extra_params,
                adapter_config,
                api_key=api_key,
            )
            config_path = Path(tmpdir) / "run_config.yaml"
            config_path.write_text(yaml.dump(run_config, sort_keys=False), encoding="utf-8")

            mounts = list(self._extra_mounts) + [f"{host}:{cont}" for host, cont in cfg_mounts.items()]

            env_vars: dict[str, str] = {}
            if api_key:
                env_vars[_API_KEY_ENV] = api_key
            env_vars.update(self._extra_env)
            env_vars.update(cfg_env_vars)

            eval_cmd = (
                "cmd=$(command -v nemo-evaluator >/dev/null 2>&1"
                " && echo nemo-evaluator || echo eval-factory)"
                f" && $cmd run_eval --run_config {_CONTAINER_CONFIG_PATH}"
            )
            inner_command = ["bash", "-c", eval_cmd]

            if _runs_on_slurm():
                if shutil.which("srun") is None:
                    raise RuntimeError(
                        "SLURM_JOB_ID is set but `srun` is not in PATH. "
                        "Either run inside an sbatch step on a SLURM cluster, "
                        "or unset SLURM_JOB_ID to force the Docker dispatch."
                    )
                logger.info(
                    "Detected SLURM_JOB_ID=%s — dispatching via Pyxis (srun --container-image)",
                    os.environ.get("SLURM_JOB_ID"),
                )
                cmd, subprocess_env = self._build_srun_cmd(results_dir, config_path, mounts, env_vars, inner_command)
            else:
                cmd, subprocess_env = self._build_docker_cmd(results_dir, config_path, mounts, env_vars, inner_command)

            logger.info("Running container: %s", _redact_cmd_for_log(cmd))
            logger.info("Command inside container: %s", " ".join(inner_command))

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=subprocess_env,
                limit=2**20,
            )

            stdout_buf: list[bytes] = []
            stderr_buf: list[bytes] = []
            log_line = lambda line: logger.info("%s | %s", self.name, line)  # noqa: E731

            async def _consume() -> None:
                await asyncio.gather(
                    _drain_stream(proc.stdout, stdout_buf, log_line),
                    _drain_stream(proc.stderr, stderr_buf, log_line),
                )
                await proc.wait()

            try:
                await asyncio.wait_for(_consume(), timeout=self._timeout)
                stdout, stderr = b"".join(stdout_buf), b"".join(stderr_buf)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout = b"".join(stdout_buf)
                stderr = b"".join(stderr_buf) + f"\ncontainer exceeded timeout of {self._timeout}s".encode()

            if proc.returncode != 0:
                logger.error(
                    "Container failed (exit %d).\n%s",
                    proc.returncode,
                    _format_container_failure(stdout, stderr),
                )

            bundle = self._parse_results(results_dir)
            # Persist the run_config the container actually consumed; the
            # tmpdir is destroyed in the finally block, so carry it in the
            # bundle for write_all() to materialize under the artifact dir
            # (mirrors the SLURM path's {out}/{safe_name}/run_config.yaml).
            bundle["_run_config"] = run_config
            return bundle
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Legacy run_config generation
    # ------------------------------------------------------------------

    def _build_legacy_run_config(
        self,
        model_url: str,
        model_id: str,
        endpoint_type: str,
        extra_params: dict[str, Any],
        adapter_config: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> dict[str, Any]:
        """Thin wrapper around :func:`build_legacy_run_config` that
        forwards this environment's task name.  Kept on the class so
        callers that already have a ``ContainerEnvironment`` instance
        don't need to pass ``task`` again.
        """
        return build_legacy_run_config(
            task=self._task,
            model_url=model_url,
            model_id=model_id,
            endpoint_type=endpoint_type,
            extra_params=extra_params,
            adapter_config=adapter_config,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Docker plumbing
    # ------------------------------------------------------------------

    def _build_docker_cmd(
        self,
        results_dir: Path,
        config_path: Path,
        mounts: list[str],
        env_vars: dict[str, str],
        inner_command: list[str],
    ) -> tuple[list[str], dict[str, str] | None]:
        """Build a complete ``docker run`` argv.

        Returns ``(argv, env)`` where ``env`` is ``None`` so the subprocess
        inherits the parent environment (env values travel inline via ``-e``).
        """
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{results_dir}:{_CONTAINER_RESULTS_DIR}",
            "-v",
            f"{config_path}:{_CONTAINER_CONFIG_PATH}:ro",
        ]
        for mount in mounts:
            cmd.extend(["-v", mount])
        for k, v in env_vars.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(self._image)
        cmd.extend(inner_command)
        return cmd, None

    def _build_srun_cmd(
        self,
        results_dir: Path,
        config_path: Path,
        mounts: list[str],
        env_vars: dict[str, str],
        inner_command: list[str],
    ) -> tuple[list[str], dict[str, str]]:
        """Build a ``srun --container-image`` argv for Pyxis/Enroot.

        Pyxis differs from Docker in flag syntax:

        - **Mounts**: a single comma-separated ``--container-mounts=H1:C1,H2:C2``
          flag instead of repeated ``-v`` args.
        - **Env vars**: ``--container-env=K1 --container-env=K2 ...`` lists
          *names only*; Pyxis inherits the values from the caller's process
          env.  We return the values as the subprocess ``env`` dict so
          :func:`asyncio.create_subprocess_exec` propagates them.

        Eval-factory containers are single-node by design (they iterate
        problems and call out to the model server over HTTP), so we hardcode
        ``--nodes=1 --ntasks=1`` and use ``--overlap`` to share the existing
        sbatch allocation.  Het-jobs and master-IP pinning are out of scope
        for this MR — see follow-up issue.

        ``--no-container-mount-home`` prevents Pyxis from mounting the
        caller's ``$HOME`` into the container; legacy eval-factory images
        run as root and could otherwise pollute the user's home dir.
        """
        all_mounts = [
            f"{results_dir}:{_CONTAINER_RESULTS_DIR}",
            f"{config_path}:{_CONTAINER_CONFIG_PATH}:ro",
            *mounts,
        ]
        cmd = [
            "srun",
            "--mpi=pmix",
            "--overlap",
            "--unbuffered",
            "--nodes=1",
            "--ntasks=1",
            f"--container-image={_resolve_pyxis_image(self._image)}",
            f"--container-mounts={','.join(all_mounts)}",
            "--no-container-mount-home",
        ]
        cmd.extend(f"--container-env={name}" for name in env_vars)
        cmd.extend(inner_command)
        return cmd, {**os.environ, **env_vars}

    # ------------------------------------------------------------------
    # Results parsing (handles both legacy and simple formats)
    # ------------------------------------------------------------------

    def _parse_results(self, results_dir: Path) -> dict[str, Any]:
        """Parse container output into NEL bundle format.

        Thin wrapper around :func:`build_legacy_bundle` so the in-process
        docker/srun path produces the same bundle shape that
        ``nel eval report`` later materialises from ``results.yml`` for
        ``--submit`` runs.
        """
        return build_legacy_bundle(self._image, self._task, results_dir)

    @staticmethod
    def _extract_scores(raw: dict[str, Any]) -> dict[str, Any]:
        """Extract scores from either legacy or simple format."""

        # Legacy Evaluator format:
        #   results:
        #     tasks:
        #       <task>:
        #         metrics:
        #           <group>:
        #             scores:
        #               <metric>:
        #                 value: 0.85
        #                 stats: { ... }
        results_block = raw.get("results")
        if isinstance(results_block, dict):
            tasks = results_block.get("tasks") or {}
            scores: dict[str, Any] = {}
            for task_name, task_data in tasks.items():
                if not isinstance(task_data, dict):
                    continue
                for group_name, group_data in (task_data.get("metrics") or {}).items():
                    if not isinstance(group_data, dict):
                        continue
                    for metric_name, metric_data in (group_data.get("scores") or {}).items():
                        if not isinstance(metric_data, dict) or "value" not in metric_data:
                            continue
                        key = f"{task_name}/{group_name}/{metric_name}"
                        scores[key] = {
                            "value": round(float(metric_data["value"]), 4),
                        }
                        if "stats" in metric_data:
                            scores[key]["stats"] = metric_data["stats"]
            if scores:
                return scores

            # Legacy format may also have groups
            groups = results_block.get("groups") or {}
            for group_name, group_data in groups.items():
                if not isinstance(group_data, dict):
                    continue
                for mg_name, mg_data in (group_data.get("metrics") or {}).items():
                    if not isinstance(mg_data, dict):
                        continue
                    for metric_name, metric_data in (mg_data.get("scores") or {}).items():
                        if not isinstance(metric_data, dict) or "value" not in metric_data:
                            continue
                        key = f"{group_name}/{mg_name}/{metric_name}"
                        scores[key] = {
                            "value": round(float(metric_data["value"]), 4),
                        }
                        if "stats" in metric_data:
                            scores[key]["stats"] = metric_data["stats"]
            if scores:
                return scores

        # Simple flat format: metrics: {acc: 0.85} or scores: {acc: {value: 0.85}}
        source = raw.get("metrics") or raw.get("scores") or {}
        scores = {}
        for key, value in source.items():
            if isinstance(value, (int, float)):
                scores[key] = {"value": round(float(value), 4)}
            elif isinstance(value, dict) and "value" in value:
                scores[key] = value
        return scores

    @staticmethod
    def _extract_sample_count(raw: dict[str, Any], scores: dict[str, Any]) -> int:
        """Extract sample count from results.

        Tries (in order): top-level ``samples`` / ``n_samples``,
        ``stats.count`` from the first scored metric, and finally
        ``config.params.limit_samples`` from the eval-factory run_config
        — needed for harnesses like lm-eval-harness that don't emit a
        per-metric count.
        """
        for key in ("samples", "n_samples"):
            val = raw.get(key)
            if val is not None:
                return int(val)
        for score_data in scores.values():
            if isinstance(score_data, dict):
                count = (score_data.get("stats") or {}).get("count")
                if count is not None:
                    return int(count)
        limit = (raw.get("config") or {}).get("params", {}).get("limit_samples")
        if isinstance(limit, int) and limit > 0:
            return limit
        return 0
