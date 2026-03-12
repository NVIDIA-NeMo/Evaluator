"""SandboxSolver: runs a command-based agent in a Docker sandbox or as a local subprocess."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import SeedResult

from .base import SolveResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


class SandboxSolver:
    """Runs an agent, either inside a per-problem sandbox or as a local subprocess.

    If ``solve()`` receives a sandbox, the agent runs inside it.  Two sandbox
    modes are supported, inferred from the sandbox spec:

    - **exec-server** (no entrypoint): solver execs the agent command inside
      the sandbox and reads output from a file.
    - **agent-server** (entrypoint starts an HTTP agent): solver connects to
      the agent API via the container IP and POSTs the task.

    If no sandbox is provided, the agent runs as a local subprocess.

    ``setup_cmd`` runs before the agent (e.g., ``pip install openhands-ai``).
    ``invocation_template`` overrides the default ``--task-file``/``--output-file``
    command protocol; supports ``{task_file}``, ``{output_file}``, ``{model_url}``,
    ``{model_id}`` placeholders, plus ``{metadata.KEY}`` for any key in the
    task's metadata dict (e.g., ``{metadata.workspace_path}``).
    """

    def __init__(
        self,
        agent_cmd: str,
        model_url: str,
        model_id: str,
        api_key: str | None = None,
        setup_cmd: str | None = None,
        invocation_template: str | None = None,
        agent_port: int = 3000,
        max_turns: int = 100,
        timeout: float = 1800.0,
    ) -> None:
        self._agent_cmd = agent_cmd
        self._model_url = model_url
        self._model_id = model_id
        self._api_key = api_key
        self._setup_cmd = setup_cmd
        self._invocation_template = invocation_template
        self._agent_port = agent_port
        self._max_turns = max_turns
        self._timeout = timeout

    def _build_command(
        self, task_file: str, output_file: str, *,
        in_sandbox: bool, metadata: dict[str, Any] | None = None,
    ) -> str:
        import shlex

        effective_model_url = (
            "$MODEL_BASE_URL" if in_sandbox else shlex.quote(self._model_url)
        )

        if self._invocation_template:
            cmd = self._invocation_template.format(
                task_file=shlex.quote(task_file),
                output_file=shlex.quote(output_file),
                model_url=effective_model_url,
                model_id=shlex.quote(self._model_id),
            )
            if metadata:
                for k, v in metadata.items():
                    cmd = cmd.replace(f"{{metadata.{k}}}", shlex.quote(str(v)))
            return cmd

        return (
            f"{self._agent_cmd} "
            f"--task-file {shlex.quote(task_file)} "
            f"--output-file {shlex.quote(output_file)} "
            f"--model-url {effective_model_url} "
            f"--model-id {shlex.quote(self._model_id)}"
        )

    async def solve(
        self, task: SeedResult, sandbox: Sandbox | None = None,
    ) -> SolveResult:
        if sandbox is not None:
            return await self._solve_sandbox(task, sandbox)
        return await self._solve_local(task)

    async def _solve_sandbox(self, task: SeedResult, sandbox: Sandbox) -> SolveResult:
        import json
        import tempfile
        from pathlib import Path

        task_data = json.dumps({
            "prompt": task.prompt,
            "expected_answer": task.expected_answer,
            "metadata": task.metadata,
        }).encode()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(task_data)
            f.flush()
            await sandbox.upload(Path(f.name), "/workspace/task.json")

        if self._setup_cmd:
            logger.info("Running agent setup: %s", self._setup_cmd[:120])
            setup_result = await sandbox.exec(self._setup_cmd, timeout_sec=600)
            if setup_result.return_code != 0:
                logger.warning("Agent setup failed (rc=%d): %s",
                               setup_result.return_code, setup_result.stderr[:500])

        if sandbox.spec.entrypoint:
            return await self._solve_agent_server(task, sandbox)

        cmd = self._build_command(
            "/workspace/task.json", "/workspace/output.json",
            in_sandbox=True, metadata=task.metadata,
        )
        result = await sandbox.exec(cmd, timeout_sec=self._timeout)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            out_path = Path(f.name)
        try:
            await sandbox.download("/workspace/output.json", out_path)
            output = json.loads(out_path.read_text())
            return SolveResult(
                response=output.get("response", ""),
                trajectory=output.get("trajectory", []),
            )
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            return SolveResult(
                response=result.stdout[:5000] if result.stdout else "[agent modified sandbox state]",
                trajectory=[{"stderr": result.stderr[:2000] if result.stderr else ""}],
            )

    async def _solve_agent_server(self, task: SeedResult, sandbox: Sandbox) -> SolveResult:
        agent_ip = sandbox.container_ip
        if not agent_ip:
            raise RuntimeError("Cannot reach agent: no container IP")
        agent_url = f"http://{agent_ip}:{self._agent_port}"

        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(f"{agent_url}/solve", json={
                "prompt": task.prompt,
                "model_url": sandbox.resolve_outside_endpoint(self._model_url),
                "model_id": self._model_id,
            })
            data = resp.json()
            return SolveResult(
                response=data.get("response", ""),
                trajectory=data.get("trajectory", []),
            )

    async def _solve_local(self, task: SeedResult) -> SolveResult:
        import asyncio
        import json
        import shlex
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory(prefix="nel_agent_") as tmpdir:
            task_file = Path(tmpdir) / "task.json"
            task_file.write_text(json.dumps({
                "prompt": task.prompt,
                "expected_answer": task.expected_answer,
                "metadata": task.metadata,
            }))

            output_file = Path(tmpdir) / "output.json"

            if self._setup_cmd:
                logger.info("Running local agent setup: %s", self._setup_cmd[:120])

            cmd = self._build_command(
                str(task_file), str(output_file),
                in_sandbox=False, metadata=task.metadata,
            )

            import os
            env = {**os.environ}
            if self._api_key:
                env["NEL_API_KEY"] = self._api_key

            process = await asyncio.create_subprocess_shell(
                f"/bin/bash -c {shlex.quote(cmd)}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return SolveResult(response="", trajectory=[{"error": "agent_timeout"}])

            if output_file.exists():
                result = json.loads(output_file.read_text())
                return SolveResult(
                    response=result.get("response", ""),
                    trajectory=result.get("trajectory", []),
                )

            return SolveResult(
                response=stdout.decode()[:5000] if stdout else "",
                trajectory=[{"stderr": stderr.decode()[:2000] if stderr else ""}],
            )

    async def close(self) -> None:
        pass
