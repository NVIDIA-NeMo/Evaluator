"""Harbor environment integration -- agent benchmarks from Harbor task directories.

Harbor tasks follow a directory layout::

    task_dir/
    ├── instruction.md          # Agent prompt
    ├── task.toml               # Config (timeouts, resources, docker_image)
    ├── environment/
    │   └── Dockerfile          # Container definition (FROM per-problem-image)
    ├── tests/
    │   └── test.sh             # Verification script (writes reward.txt)
    └── solution/               # Optional reference solution
        └── solve.sh

The environment loads a directory of such tasks and maps them into NEL's
seed/verify contract.  Each task gets its own sandbox with the per-problem
Docker image specified in ``task.toml`` (field ``environment.docker_image``)
or parsed from the Dockerfile's ``FROM`` line as a fallback.
"""
from __future__ import annotations

import logging
import re
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox, SandboxSpec

logger = logging.getLogger(__name__)


def _parse_docker_image_from_toml(task_toml: Path) -> str | None:
    """Read ``environment.docker_image`` from task.toml."""
    try:
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        return config.get("environment", {}).get("docker_image")
    except Exception:
        return None


def _parse_docker_image_from_dockerfile(dockerfile: Path) -> str | None:
    """Extract image from the first ``FROM`` line of a Dockerfile."""
    try:
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FROM "):
                parts = stripped.split()
                if len(parts) >= 2:
                    image = parts[1]
                    if image.startswith("$") or image.lower() == "scratch":
                        return None
                    return image.split(" AS ")[0].split(" as ")[0].strip()
    except Exception:
        pass
    return None


def _parse_task_config(task_toml: Path) -> dict[str, Any]:
    """Parse task.toml and return the full config dict."""
    try:
        return tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return {}


class HarborEnvironment(EvalEnvironment):
    """Wraps a directory of Harbor tasks as an EvalEnvironment.

    Each subdirectory that contains ``instruction.md`` is treated as a task.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        num_examples: int | None = None,
    ) -> None:
        super().__init__()
        self._dataset_path = Path(dataset_path)
        self._tasks: list[Path] = []
        self.name = self._dataset_path.name

        if not self._dataset_path.is_dir():
            raise FileNotFoundError(f"Harbor dataset directory not found: {self._dataset_path}")

        for task_dir in sorted(self._dataset_path.iterdir()):
            if task_dir.is_dir() and (task_dir / "instruction.md").exists():
                self._tasks.append(task_dir)

        if num_examples is not None:
            self._tasks = self._tasks[:num_examples]

        logger.info("Harbor %s: %d tasks loaded from %s",
                     self.name, len(self._tasks), self._dataset_path)

    async def dataset_size(self) -> int:
        return len(self._tasks)

    def _resolve_image(self, task_dir: Path) -> str | None:
        """Resolve Docker image for a task: task.toml first, Dockerfile fallback."""
        task_toml = task_dir / "task.toml"
        if task_toml.exists():
            image = _parse_docker_image_from_toml(task_toml)
            if image:
                return image

        dockerfile = task_dir / "environment" / "Dockerfile"
        if dockerfile.exists():
            return _parse_docker_image_from_dockerfile(dockerfile)

        return None

    async def seed(self, idx: int) -> SeedResult:
        task_dir = self._tasks[idx]
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()

        task_toml = task_dir / "task.toml"
        config = _parse_task_config(task_toml) if task_toml.exists() else {}

        image = self._resolve_image(task_dir)

        from nemo_evaluator.sandbox.base import SandboxSpec

        sandbox_spec = None
        verify_sandbox_spec = None
        capture_cmd = None
        apply_cmd = None
        if image:
            sandbox_spec = SandboxSpec(
                image=image,
                workdir="/testbed",
                env={"HARBOR_TASK_DIR": str(task_dir)},
            )
            verify_sandbox_spec = SandboxSpec(
                image=image,
                workdir="/testbed",
            )
            capture_cmd = "cd /testbed && tar cf /output/workspace.tar --exclude=.git ."
            apply_cmd = (
                "if [ -f /input/workspace.tar ]; then "
                "tar xf /input/workspace.tar -C /testbed; fi"
            )

        metadata: dict[str, Any] = {
            "source": "harbor",
            "task_id": task_dir.name,
            "task_dir": str(task_dir),
        }
        task_metadata = config.get("metadata", {})
        if task_metadata:
            metadata.update(task_metadata)

        return SeedResult(
            prompt=instruction,
            expected_answer="",
            metadata=metadata,
            sandbox_spec=sandbox_spec,
            verify_sandbox_spec=verify_sandbox_spec,
            capture_cmd=capture_cmd,
            apply_cmd=apply_cmd,
        )

    async def verify(
        self, response: str, expected: str,
        sandbox: Sandbox | None = None, **metadata: Any,
    ) -> VerifyResult:
        """Run test scripts in the verification sandbox.

        With the StatelessSandbox lifecycle the workspace has already been
        restored from ``/input/workspace.tar`` by ``apply_cmd``.  This
        method only uploads test scripts and executes them.
        """
        if sandbox is None:
            logger.warning("Harbor verify called without sandbox -- cannot run tests")
            return VerifyResult(
                reward=0.0,
                scoring_details={"method": "harbor", "error": "no_sandbox"},
            )

        task_dir_str = metadata.get("task_dir")
        if not task_dir_str:
            return VerifyResult(
                reward=0.0,
                scoring_details={"method": "harbor", "error": "missing_task_dir"},
            )

        task_dir = Path(task_dir_str)
        tests_dir = task_dir / "tests"

        if not tests_dir.exists():
            return VerifyResult(
                reward=0.0,
                scoring_details={"method": "harbor", "error": "no_tests_dir"},
            )

        for test_file in sorted(tests_dir.rglob("*")):
            if test_file.is_file():
                rel = test_file.relative_to(tests_dir)
                await sandbox.upload(test_file, f"/tests/{rel}")

        await sandbox.exec("chmod +x /tests/test.sh", timeout_sec=10)
        result = await sandbox.exec("bash /tests/test.sh", timeout_sec=600)

        reward = 0.0
        reward_details: dict[str, Any] = {
            "method": "harbor",
            "test_exit_code": result.return_code,
            "test_stdout": result.stdout[:2000] if result.stdout else "",
            "test_stderr": result.stderr[:2000] if result.stderr else "",
        }

        import tempfile

        try:
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                reward_path = Path(f.name)
            await sandbox.download("/logs/verifier/reward.txt", reward_path)
            reward_text = reward_path.read_text(encoding="utf-8").strip()
            reward = float(reward_text)
            reward_details["reward_raw"] = reward_text
        except (FileNotFoundError, OSError):
            try:
                with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                    reward_path = Path(f.name)
                await sandbox.download("/reward.txt", reward_path)
                reward_text = reward_path.read_text(encoding="utf-8").strip()
                reward = float(reward_text)
                reward_details["reward_raw"] = reward_text
            except (FileNotFoundError, OSError, ValueError):
                reward = 1.0 if result.return_code == 0 else 0.0
                reward_details["reward_source"] = "exit_code_fallback"
        except ValueError:
            reward = 1.0 if result.return_code == 0 else 0.0
            reward_details["reward_source"] = "exit_code_fallback"

        return VerifyResult(
            reward=reward,
            scoring_details=reward_details,
        )

    async def close(self) -> None:
        pass
