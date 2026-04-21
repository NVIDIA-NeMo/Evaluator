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
"""NMP Harbor -- agentic-use tasks + auto-built ``nmp-harbor`` base image.

Convenience wrapper around :class:`HarborEnvironment` for NVIDIA NMP's
agentic-use Harbor suite.  Every per-task Dockerfile in that suite is
``FROM nmp-harbor:latest``, an image built from ``Dockerfile.harbor``
inside the NMP repo (not published to a registry).

This benchmark:

1. Reads ``NMP_REPO`` (or the ``nmp_repo`` param) to locate the NMP checkout.
2. Prepends an :class:`ImageBuildRequest` that builds ``nmp-harbor:latest``
   from ``$NMP_REPO/Dockerfile.harbor`` on first use.
3. Delegates task discovery + verification to :class:`HarborEnvironment`
   against ``$NMP_REPO/tests/agentic-use``.
4. Forces stateful sandbox mode -- NMP state lives in the running API
   process, so the verifier must reuse the agent container.  We clear
   ``verify_sandbox_spec`` in :meth:`seed` and nel-next's lifecycle
   picker selects :class:`StatefulSandbox`.
5. Before the verifier runs, ``docker exec``s a readiness probe against
   ``http://localhost:8080/health/ready``.  ``docker run -d`` returns
   before the image ENTRYPOINT's ``nemo services run`` binds :8080, so
   without this gate the grader races API startup with ``Connection
   refused``.  Pass ``params.wait_api_cmd=""`` to skip.

See ``examples/configs/10_nmp_harbor.yaml`` for a ready-to-run config.
"""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from nemo_evaluator.environments.base import SeedResult, VerifyResult
from nemo_evaluator.environments.harbor import HarborEnvironment
from nemo_evaluator.environments.registry import register
from nemo_evaluator.sandbox.base import ImageBuildRequest, ImageSpec, Sandbox

logger = logging.getLogger(__name__)


_NMP_API_LOG = "/tmp/nmp-api.log"
_HEALTH_URL = "http://localhost:8080/health/ready"

# Readiness probe used by :meth:`NmpHarborEnvironment.verify`.
#
# Targets ``/health/ready`` (NB: the image's built-in probe uses
# ``/health``, which is a 404 -- it "passes" because ``curl -s`` without
# ``-f`` exits 0 on any HTTP response).  On failure, dumps enough
# diagnostics (listening sockets, api process tree, api log tail,
# verbose curl) to explain why the endpoint is unreachable or unready.
_WAIT_API_CMD = rf"""set +e
echo 'nmp_harbor: probing {_HEALTH_URL}' >&2
SUCC=0
LAST=''
# 60 iterations x (2s curl timeout + 1s sleep) -> <=180s worst case;
# in practice the image ENTRYPOINT makes the API healthy within ~15s.
for _ in $(seq 1 60); do
    LAST=$(curl -sS -m 2 -o /dev/null -w '%{{http_code}}' {_HEALTH_URL} 2>&1)
    if [ "$LAST" = "200" ]; then
        SUCC=$((SUCC+1))
        [ "$SUCC" -ge 3 ] && break
    else
        SUCC=0
    fi
    sleep 1
done

if [ "$SUCC" -ge 3 ]; then
    echo 'nmp_harbor: NMP API ready on :8080' >&2
    exit 0
fi

{{
    echo 'nmp_harbor: NMP API failed to become ready on :8080'
    echo "  last curl result: $LAST"
    echo '--- id ---'
    id || true
    echo '--- listening sockets ---'
    (ss -lntp 2>&1 || netstat -lntp 2>&1 || true) | head -40
    echo '--- api-related processes ---'
    ps -eo pid,ppid,user,args 2>&1 | grep -iE 'nemo|uvicorn|python' | grep -v grep || true
    echo '--- tail {_NMP_API_LOG} ---'
    tail -n 200 {_NMP_API_LOG} 2>&1 || echo '(no api log)'
    echo '--- curl -v {_HEALTH_URL} ---'
    curl -v -m 5 {_HEALTH_URL} 2>&1 || true
}} >&2
exit 1
"""


def _resolve_nmp_repo(nmp_repo: str | None) -> Path:
    candidate = nmp_repo or os.environ.get("NMP_REPO")
    if not candidate:
        raise ValueError(
            "nmp_harbor: set the NMP_REPO environment variable or pass "
            "nmp_repo= via benchmark params (path to a local NMP checkout)."
        )
    repo = Path(candidate).expanduser().resolve()
    if not repo.is_dir():
        raise ValueError(f"nmp_harbor: NMP_REPO path does not exist: {repo}")
    return repo


@register("nmp_harbor")
class NmpHarborEnvironment(HarborEnvironment):
    """NMP agentic-use Harbor tasks with an auto-built ``nmp-harbor`` base image.

    Parameters
    ----------
    nmp_repo:
        Path to a local NMP checkout.  Falls back to ``$NMP_REPO``.
    dataset_subdir:
        Harbor tasks directory relative to ``nmp_repo``.
    base_image, base_dockerfile:
        Image tag and Dockerfile (relative to ``nmp_repo``) built once per run.
    task_names:
        Restrict to one or more task names (exact match).  Accepts a list or
        a single scalar string (for CLI ``-O`` compatibility).
    num_examples:
        Additional cap on the task list after ``task_names`` filtering.
    wait_api_cmd:
        Shell command exec'd before the Harbor verifier to gate on API
        readiness.  Pass an empty string to disable.
    """

    def __init__(
        self,
        nmp_repo: str | None = None,
        dataset_subdir: str = "tests/agentic-use",
        base_image: str = "nmp-harbor:latest",
        base_dockerfile: str = "Dockerfile.harbor",
        task_names: str | list[str] | None = None,
        num_examples: int | None = None,
        wait_api_cmd: str | None = None,
    ) -> None:
        repo = _resolve_nmp_repo(nmp_repo)

        dataset_path = repo / dataset_subdir
        if not dataset_path.is_dir():
            raise ValueError(f"nmp_harbor: dataset directory not found: {dataset_path} (override with dataset_subdir=)")

        dockerfile_path = repo / base_dockerfile
        if not dockerfile_path.is_file():
            raise ValueError(
                f"nmp_harbor: base Dockerfile not found: {dockerfile_path} (override with base_dockerfile=)"
            )

        super().__init__(dataset_path=dataset_path, num_examples=None)

        self._tasks = self._filter_tasks(task_names, num_examples)

        self._base_image = base_image
        self._base_dockerfile = dockerfile_path
        self._base_context = repo
        self._wait_api_cmd = _WAIT_API_CMD if wait_api_cmd is None else wait_api_cmd

    def _filter_tasks(
        self,
        task_names: str | list[str] | None,
        num_examples: int | None,
    ) -> list[Any]:
        tasks = self._tasks
        if task_names:
            wanted = {task_names} if isinstance(task_names, str) else set(task_names)
            tasks = [t for t in tasks if t.name in wanted]
            missing = wanted - {t.name for t in tasks}
            if missing:
                raise ValueError(f"nmp_harbor: task_names not found under {self._dataset_path}: {sorted(missing)}")
            logger.info(
                "nmp_harbor: filtered to %d task(s) by task_names: %s",
                len(tasks),
                [t.name for t in tasks],
            )
        if num_examples is not None:
            tasks = tasks[:num_examples]
        return tasks

    async def seed(self, idx: int) -> SeedResult:
        result = await super().seed(idx)
        result.verify_sandbox_spec = None
        result.capture_cmd = None
        result.apply_cmd = None
        return result

    async def verify(
        self,
        response: str,
        expected: str,
        sandbox: Sandbox | None = None,
        **metadata: Any,
    ) -> VerifyResult:
        """Gate on NMP API readiness, then delegate to the Harbor verifier.

        ``docker run -d`` returns before the image ENTRYPOINT's background
        ``nemo services run`` has bound ``:8080``, so without this gate
        the grader fails almost immediately with ``Connection refused``.
        """
        if sandbox is not None and self._wait_api_cmd:
            logger.info("nmp_harbor: waiting for NMP API readiness in verify sandbox")
            wait = await sandbox.exec(self._wait_api_cmd, timeout_sec=300)
            if wait.return_code != 0:
                return self._api_not_ready_result(wait)
        return await super().verify(response, expected, sandbox=sandbox, **metadata)

    @staticmethod
    def _api_not_ready_result(wait: Any) -> VerifyResult:
        stdout = (wait.stdout or "")[-4000:]
        stderr = (wait.stderr or "")[-4000:]
        logger.error(
            "nmp_harbor: NMP API readiness probe failed (rc=%d)\nstdout:\n%s\nstderr:\n%s",
            wait.return_code,
            stdout,
            stderr,
        )
        return VerifyResult(
            reward=0.0,
            extracted_answer="nmp_api_not_ready",
            scoring_details={
                "method": "harbor",
                "error": "nmp_api_not_ready",
                "wait_stderr": stderr,
                "wait_stdout": stdout,
            },
        )

    async def image_build_requests(self) -> list[Any] | None:
        base_spec = ImageSpec(
            image=self._base_image,
            source={
                "dockerfile": str(self._base_dockerfile),
                "context": str(self._base_context),
            },
        )
        base_req = ImageBuildRequest(specs=[base_spec], docker_build_fn=_docker_build_base)
        task_reqs = await super().image_build_requests() or []
        return [base_req, *task_reqs]


def _docker_build_base(missing: list[ImageSpec]) -> None:
    """Build each missing base image via the local Docker daemon."""
    for spec in missing:
        src = spec.source or {}
        dockerfile = src.get("dockerfile")
        context = src.get("context")
        if not dockerfile or not context:
            raise RuntimeError(f"nmp_harbor: missing build context for {spec.image}")
        logger.info("nmp_harbor: building %s from %s", spec.image, dockerfile)
        subprocess.run(
            ["docker", "build", "-t", spec.image, "-f", dockerfile, context],
            check=True,
        )
