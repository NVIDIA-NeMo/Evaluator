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
"""Harbor environment integration -- agent benchmarks from Harbor task directories.

Uses Harbor's native registry to discover and download benchmark tasks via
sparse git checkout.  Supports all datasets in the Harbor registry (60+
benchmarks including terminal-bench, swebench, usaco, bird-bench, etc.).

Harbor tasks follow a directory layout::

    task_dir/
    ├── instruction.md          # Agent prompt
    ├── task.toml               # Config (timeouts, resources, docker_image)
    ├── environment/
    │   └── Dockerfile          # Container definition
    ├── tests/
    │   └── test.sh             # Verification script (writes reward.txt)
    └── solution/               # Optional reference solution

Auto-download
~~~~~~~~~~~~~
When ``harbor://<name>`` (or ``harbor://<name>@<version>``) is used and the
local dataset directory doesn't exist, :func:`auto_prepare` looks up the
Harbor registry and downloads the task directories via sparse git checkout.

Registry resolution order:

1. ``HARBOR_REGISTRY`` env-var (path to a local ``registry.json``)
2. Sibling ``harbor/`` repo relative to the workspace root
3. Download from GitHub (``laude-institute/harbor/main/registry.json``)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

REGISTRY_URL = "https://raw.githubusercontent.com/laude-institute/harbor/main/registry.json"

# ---------------------------------------------------------------------------
# Registry data model
# ---------------------------------------------------------------------------


@dataclass
class RegistryTask:
    name: str
    git_url: str
    git_commit_id: str | None
    path: str


@dataclass
class DatasetSpec:
    name: str
    version: str
    description: str
    tasks: list[RegistryTask]


# ---------------------------------------------------------------------------
# Registry loading + caching
# ---------------------------------------------------------------------------

_registry_cache: list[DatasetSpec] | None = None


def _locate_registry_json() -> Path | None:
    """Search common locations for a local ``registry.json``."""
    env = os.environ.get("HARBOR_REGISTRY")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    src_root = Path(__file__).resolve().parents[4]  # …/agentic-party
    candidate = src_root / "harbor" / "registry.json"
    if candidate.is_file():
        return candidate

    cwd_candidate = Path.cwd()
    for _ in range(5):
        c = cwd_candidate / "harbor" / "registry.json"
        if c.is_file():
            return c
        cwd_candidate = cwd_candidate.parent

    return None


def _parse_raw(raw: list[dict]) -> list[DatasetSpec]:
    return [
        DatasetSpec(
            name=d["name"],
            version=d["version"],
            description=d.get("description", ""),
            tasks=[
                RegistryTask(
                    name=t["name"],
                    git_url=t["git_url"],
                    git_commit_id=t.get("git_commit_id"),
                    path=t["path"],
                )
                for t in d.get("tasks", [])
            ],
        )
        for d in raw
    ]


def _load_registry() -> list[DatasetSpec]:
    local = _locate_registry_json()
    if local:
        logger.info("Loading Harbor registry from %s", local)
        return _parse_raw(json.loads(local.read_text()))

    logger.info("Downloading Harbor registry from %s", REGISTRY_URL)
    import json as _json
    import urllib.request

    with urllib.request.urlopen(REGISTRY_URL, timeout=60) as resp:
        return _parse_raw(_json.loads(resp.read()))


def get_registry() -> list[DatasetSpec]:
    global _registry_cache
    if _registry_cache is None:
        _registry_cache = _load_registry()
        logger.info("Harbor registry: %d datasets loaded", len(_registry_cache))
    return _registry_cache


def find_dataset(name: str, version: str | None = None) -> DatasetSpec:
    """Find a dataset by name and optional version.

    Raises :class:`KeyError` when not found.
    """
    registry = get_registry()
    matches = [d for d in registry if d.name == name]
    if not matches:
        available = sorted({d.name for d in registry})
        raise KeyError(
            f"Harbor dataset {name!r} not found. Available ({len(available)}): {', '.join(available[:20])} …"
        )

    if version:
        for m in matches:
            if m.version == version:
                return m
        versions = [m.version for m in matches]
        raise KeyError(f"Harbor dataset {name!r} version {version!r} not found. Available versions: {versions}")

    # Auto-resolve best version
    try:
        from packaging.version import InvalidVersion, Version

        semver = []
        for m in matches:
            if m.version == "head":
                return m
            try:
                semver.append((Version(m.version), m))
            except InvalidVersion:
                pass
        if semver:
            semver.sort(key=lambda x: x[0], reverse=True)
            return semver[0][1]
    except ImportError:
        pass

    return matches[-1]


# ---------------------------------------------------------------------------
# Task downloader (sparse git checkout)
# ---------------------------------------------------------------------------


def download_harbor_tasks(
    dataset: DatasetSpec,
    output_dir: Path,
    limit: int | None = None,
) -> Path:
    """Download Harbor tasks via sparse git checkout.

    Tasks that already exist on disk are skipped.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = dataset.tasks
    if limit:
        tasks = tasks[:limit]

    cached = {d.name for d in output_dir.iterdir() if d.is_dir() and (d / "instruction.md").exists()}
    needed = [t for t in tasks if t.name not in cached]
    if not needed:
        logger.info(
            "Harbor tasks already present: %d/%d in %s",
            len(cached),
            len(tasks),
            output_dir,
        )
        return output_dir

    logger.info(
        "Harbor cache has %d/%d tasks; downloading %d missing",
        len(cached),
        len(tasks),
        len(needed),
    )
    tasks = needed

    groups: dict[tuple[str, str | None], list[RegistryTask]] = defaultdict(list)
    for task in tasks:
        groups[(task.git_url, task.git_commit_id)].append(task)

    logger.info(
        "Downloading %d tasks from %d git source(s) for %s@%s",
        len(tasks),
        len(groups),
        dataset.name,
        dataset.version,
    )

    for (git_url, commit_id), group_tasks in groups.items():
        _download_task_group(git_url, commit_id, group_tasks, output_dir)

    n = sum(1 for d in output_dir.iterdir() if d.is_dir() and (d / "instruction.md").exists())
    logger.info("Harbor download complete: %d tasks in %s", n, output_dir)
    return output_dir


def _git(
    cmd: list[str],
    *,
    input: str | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with logging on failure."""
    try:
        return subprocess.run(
            cmd,
            input=input,
            text=True,
            encoding="utf-8",
            check=True,
            capture_output=True,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as exc:
        logger.error(
            "git command failed: %s\nstdout: %s\nstderr: %s",
            " ".join(cmd),
            exc.stdout,
            exc.stderr,
        )
        raise


def _download_task_group(
    git_url: str,
    commit_id: str | None,
    tasks: list[RegistryTask],
    output_dir: Path,
) -> None:
    """Sparse-checkout a batch of tasks from the same git repo + commit."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        logger.info("Cloning %s (sparse, depth=1)…", git_url)
        _git(
            ["git", "clone", "--filter=blob:none", "--depth", "1", "--no-checkout", git_url, str(tmp_dir)],
        )

        sparse_paths = [t.path for t in tasks]
        _git(
            ["git", "sparse-checkout", "set", "--no-cone", "--stdin"],
            input="\n".join(sparse_paths),
            cwd=tmp_dir,
        )

        if commit_id and commit_id.upper() != "HEAD":
            _git(
                ["git", "fetch", "--depth", "1", "origin", commit_id],
                cwd=tmp_dir,
            )
            _git(["git", "checkout", commit_id], cwd=tmp_dir)
        else:
            _git(["git", "checkout"], cwd=tmp_dir)

        for task in tasks:
            src = tmp_dir / task.path
            if not src.is_dir():
                logger.warning("Task path not found after checkout: %s (skipped)", src)
                continue
            dst = output_dir / task.name
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# auto_prepare  (called from registry.py)
# ---------------------------------------------------------------------------


def auto_prepare(name: str, output_dir: Path, limit: int | None = None) -> bool:
    """If *name* is in the Harbor registry, download tasks.  Returns True on success."""
    version: str | None = None
    if "@" in name:
        name, version = name.rsplit("@", 1)

    try:
        dataset = find_dataset(name, version)
    except KeyError as exc:
        logger.debug("Harbor auto_prepare: %s", exc)
        return False

    download_harbor_tasks(dataset, output_dir, limit=limit)
    return True


# ---------------------------------------------------------------------------
# Helpers for HarborEnvironment
# ---------------------------------------------------------------------------


def _parse_docker_image_from_toml(task_toml: Path) -> str | None:
    try:
        config = tomllib.loads(task_toml.read_text(encoding="utf-8"))
        return config.get("environment", {}).get("docker_image")
    except Exception:
        return None


def _parse_from_image(dockerfile: Path) -> str | None:
    """Extract the base image from the first ``FROM`` line of a Dockerfile."""
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


def _parse_workdir(dockerfile: Path, default: str = "/testbed") -> str:
    """Extract the last ``WORKDIR`` directive from a Dockerfile."""
    workdir = default
    try:
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("WORKDIR "):
                parts = stripped.split(None, 1)
                if len(parts) == 2 and not parts[1].startswith("$"):
                    workdir = parts[1].strip()
    except Exception:
        pass
    return workdir


def _dockerfile_has_extra_layers(dockerfile: Path) -> bool:
    """True when the Dockerfile has ``RUN``/``COPY``/``ADD`` beyond the FROM."""
    try:
        for line in dockerfile.read_text(encoding="utf-8").splitlines():
            stripped = line.strip().upper()
            if stripped.startswith(("RUN ", "COPY ", "ADD ")):
                return True
    except Exception:
        pass
    return False


def _content_hash_env(env_dir: Path) -> str:
    """SHA-256 of all files in an environment directory (Dockerfile, scripts, etc.).

    Sorted by relative path for determinism.  Returns first 12 hex chars.
    """
    import hashlib as _hl

    h = _hl.sha256()
    for p in sorted(env_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(env_dir).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _built_image_tag(dataset_name: str, task_name: str, env_dir: Path | None = None) -> str:
    safe_ds = dataset_name.replace("/", "__").replace("@", "_")
    safe_task = task_name.replace("/", "__").replace(":", "_")
    tag = _content_hash_env(env_dir) if env_dir and env_dir.is_dir() else "latest"
    return f"nel-harbor/{safe_ds}/{safe_task}:{tag}"


def _parse_task_config(task_toml: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(task_toml.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_harbor_dockerfiles(
    specs: list[Any],
    build_contexts: dict[str, Path],
) -> None:
    """Build Docker images from Harbor task Dockerfiles."""
    import subprocess as _sp

    for spec in specs:
        ctx = build_contexts.get(spec.image)
        if ctx is None:
            logger.warning("No build context for %s — skipping", spec.image)
            continue
        dockerfile = ctx / "Dockerfile"
        if not dockerfile.exists():
            logger.warning("Dockerfile missing: %s — skipping", dockerfile)
            continue

        logger.info("Building Harbor image: %s", spec.image)
        result = _sp.run(
            ["docker", "build", "-t", spec.image, "-f", str(dockerfile), str(ctx)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            logger.error(
                "Docker build failed for %s:\n%s",
                spec.image,
                result.stderr[-2000:],
            )
            raise RuntimeError(f"Docker build failed for {spec.image}")


# ---------------------------------------------------------------------------
# Verify helpers
# ---------------------------------------------------------------------------


async def _download_json(sandbox: Any, remote_path: str) -> dict | None:
    """Download a JSON file from the sandbox, return parsed dict or None."""
    import tempfile as _tf

    try:
        with _tf.NamedTemporaryFile(suffix=".json", delete=False) as f:
            local_path = Path(f.name)
        await sandbox.download(remote_path, local_path)
        return json.loads(local_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    finally:
        try:
            local_path.unlink(missing_ok=True)
        except Exception:
            pass


def _extract_test_section(stdout: str) -> str:
    """Pull the meaningful test/grading section out of raw test.sh stdout.

    Strips pip-install and git-checkout preamble by looking for known markers
    (test runner banners, pytest output patterns, SWEBench grading output).
    Falls back to the last 4000 chars if no markers found.
    """
    _MARKERS = [
        "PASSED",
        "FAILED",
        "ERROR",
        "SWEBench results starts here",
        "test session starts",
        "======",
        "------",
        "Ran ",
        " ... ok",
        " ... FAIL",
    ]
    lines = stdout.splitlines()
    first_interesting = len(lines)
    for i, line in enumerate(lines):
        if any(m in line for m in _MARKERS):
            first_interesting = i
            break

    if first_interesting < len(lines):
        return "\n".join(lines[first_interesting:])[-8000:]

    return stdout[-4000:]


_PROGRAMMING_LANGUAGES = frozenset(
    {
        "python",
        "java",
        "javascript",
        "typescript",
        "go",
        "rust",
        "c",
        "cpp",
        "php",
        "ruby",
        "scala",
        "kotlin",
        "swift",
    }
)


def _infer_repo_language(task_dir: Path, metadata: dict[str, Any]) -> None:
    """Populate ``metadata["repo_language"]`` from available signals.

    1. ``repo_language`` / ``language`` in ``tests/config.json`` (Pro dataset).
    2. Language-valued tag in ``task.toml [metadata] tags`` (Multilingual adapter).
    """
    config_json = task_dir / "tests" / "config.json"
    if config_json.is_file():
        try:
            raw = json.loads(config_json.read_text(encoding="utf-8"))
            lang = raw.get("repo_language") or raw.get("language")
            if lang:
                metadata["repo_language"] = lang
                return
        except Exception:
            pass

    tags = metadata.get("tags")
    if isinstance(tags, list):
        for tag in reversed(tags):
            if isinstance(tag, str) and tag.lower() in _PROGRAMMING_LANGUAGES:
                metadata["repo_language"] = tag
                return


# ---------------------------------------------------------------------------
# HarborEnvironment
# ---------------------------------------------------------------------------


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

        logger.info(
            "Harbor %s: %d tasks loaded from %s",
            self.name,
            len(self._tasks),
            self._dataset_path,
        )

    async def dataset_size(self) -> int:
        return len(self._tasks)

    async def image_build_requests(self) -> list[Any] | None:
        from nemo_evaluator.sandbox.base import ImageBuildRequest, ImageSpec

        specs: list[ImageSpec] = []
        build_contexts: dict[str, Path] = {}

        for task_dir in self._tasks:
            env_dir = task_dir / "environment"
            dockerfile = env_dir / "Dockerfile"
            if dockerfile.exists() and _dockerfile_has_extra_layers(dockerfile):
                tag = _built_image_tag(self.name, task_dir.name, env_dir)
                specs.append(ImageSpec(image=tag, source={"task_dir": str(task_dir)}))
                build_contexts[tag] = env_dir

        if not specs:
            return None

        def _docker_build(missing: list[ImageSpec]) -> None:
            _build_harbor_dockerfiles(missing, build_contexts)

        logger.info("Harbor %s: %d tasks need Dockerfile builds", self.name, len(specs))
        return [ImageBuildRequest(specs=specs, docker_build_fn=_docker_build)]

    def _resolve_image(self, task_dir: Path) -> str | None:
        """Resolve Docker image for a task.

        Priority:
        1. ``task.toml [environment] docker_image``
        2. Dockerfile with extra layers → built image tag
        3. Dockerfile FROM-only → base image directly
        """
        task_toml = task_dir / "task.toml"
        if task_toml.exists():
            image = _parse_docker_image_from_toml(task_toml)
            if image:
                return image

        env_dir = task_dir / "environment"
        dockerfile = env_dir / "Dockerfile"
        if dockerfile.exists():
            if _dockerfile_has_extra_layers(dockerfile):
                return _built_image_tag(self.name, task_dir.name, env_dir)
            return _parse_from_image(dockerfile)

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
            env_dir = task_dir / "environment"
            dockerfile = env_dir / "Dockerfile"
            wd = (
                config.get("environment", {}).get("workdir")
                or (_parse_workdir(dockerfile) if dockerfile.exists() else None)
                or "/testbed"
            )
            sandbox_spec = SandboxSpec(
                image=image,
                workdir=wd,
                env={"HARBOR_TASK_DIR": str(task_dir)},
                environment_dir=str(env_dir) if env_dir.is_dir() else None,
            )
            verify_sandbox_spec = SandboxSpec(
                image=image,
                workdir=wd,
                environment_dir=str(env_dir) if env_dir.is_dir() else None,
            )
            capture_cmd = (
                f"cd {wd} && "
                "_NEL_OUT=/output; "
                '[ -n "$_NEL_EFS_SESSION" ] && _NEL_OUT="/output/$_NEL_EFS_SESSION"; '
                "mkdir -p $_NEL_OUT && "
                "if [ -d .git ] && git rev-parse HEAD >/dev/null 2>&1; then "
                "_NEL_BASE=$(cat /tmp/_nel_base_commit 2>/dev/null || git rev-parse HEAD); "
                "_NEL_HEAD=$(git rev-parse HEAD 2>/dev/null); "
                'echo "base=$_NEL_BASE head=$_NEL_HEAD" > $_NEL_OUT/_nel_capture_info.txt; '
                "git log --oneline -3 >> $_NEL_OUT/_nel_capture_info.txt 2>/dev/null; "
                "git add -A 2>/dev/null; "
                "git diff --cached --binary $_NEL_BASE > $_NEL_OUT/_nel_patch.diff 2>/dev/null && "
                "echo git-diff > $_NEL_OUT/_nel_mode && "
                "cd $_NEL_OUT && tar cf workspace.tar _nel_patch.diff _nel_mode _nel_capture_info.txt || "
                f"{{ cd {wd} && tar cf $_NEL_OUT/workspace.tar --exclude=.git .; }}; "
                "else "
                "tar cf $_NEL_OUT/workspace.tar --exclude=.git .; "
                "fi"
            )
            apply_cmd = (
                "mkdir -p /tmp/_nel_ws && "
                "_NEL_TAR=/input/workspace.tar; "
                '[ -n "$_NEL_EFS_SESSION" ] && [ -f "/input/$_NEL_EFS_SESSION/workspace.tar" ] && '
                '_NEL_TAR="/input/$_NEL_EFS_SESSION/workspace.tar"; '
                "if [ -f $_NEL_TAR ]; then "
                "tar xf $_NEL_TAR -C /tmp/_nel_ws 2>/dev/null; "
                "[ -f /tmp/_nel_ws/_nel_capture_info.txt ] && "
                "cat /tmp/_nel_ws/_nel_capture_info.txt >&2; "
                "if [ -f /tmp/_nel_ws/_nel_patch.diff ]; then "
                "if [ -s /tmp/_nel_ws/_nel_patch.diff ]; then "
                f"cd {wd} && {{ "
                "git apply --binary --whitespace=fix /tmp/_nel_ws/_nel_patch.diff 2>/dev/null || "
                "git apply --binary --reject --whitespace=fix /tmp/_nel_ws/_nel_patch.diff 2>&1 || "
                "{ echo 'NEL: --reject failed, trying 3-way merge' >&2 && "
                "git apply --binary --3way /tmp/_nel_ws/_nel_patch.diff 2>&1; }; "
                "}; "
                "fi; "
                "else "
                f"tar xf $_NEL_TAR -C {wd}; "
                "fi; "
                "fi"
            )

        metadata: dict[str, Any] = {
            "source": "harbor",
            "task_id": task_dir.name,
            "task_dir": str(task_dir),
        }
        task_metadata = config.get("metadata", {})
        if task_metadata:
            metadata.update(task_metadata)

        # Infer repo_language when missing from task.toml metadata.
        if "repo_language" not in metadata:
            _infer_repo_language(task_dir, metadata)

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
        self,
        response: str,
        expected: str,
        sandbox: Sandbox | None = None,
        **metadata: Any,
    ) -> VerifyResult:
        """Run test scripts in the verification sandbox."""
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

        await sandbox.exec("mkdir -p /tests /logs/verifier", timeout_sec=10)

        for test_file in sorted(tests_dir.rglob("*")):
            if test_file.is_file():
                rel = test_file.relative_to(tests_dir)
                parent = str(Path(f"/tests/{rel}").parent)
                if parent != "/tests":
                    await sandbox.exec(f"mkdir -p {parent}", timeout_sec=10)
                await sandbox.upload(test_file, f"/tests/{rel}")

        await sandbox.exec("chmod -R +x /tests/", timeout_sec=10)
        result = await sandbox.exec(
            'export PATH="/root/.local/bin:/root/.cargo/bin:/usr/local/go/bin'
            ":/usr/local/cargo/bin:$HOME/.local/bin:$HOME/.cargo/bin"
            ':$HOME/go/bin:$JAVA_HOME/bin:$PATH" && '
            "bash /tests/test.sh",
            timeout_sec=600,
        )

        reward = 0.0
        _MAX_LOG = 50_000
        full_stdout = result.stdout or ""
        full_stderr = result.stderr or ""

        test_summary = _extract_test_section(full_stdout)

        reward_details: dict[str, Any] = {
            "method": "harbor",
            "test_exit_code": result.return_code,
            "test_summary": test_summary,
            "test_stdout": full_stdout[-_MAX_LOG:],
            "test_stderr": full_stderr[-_MAX_LOG:],
        }

        import tempfile as _tempfile

        report = await _download_json(sandbox, "/logs/verifier/report.json")
        if report is not None:
            reward_details["report"] = report

        try:
            with _tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                reward_path = Path(f.name)
            await sandbox.download("/logs/verifier/reward.txt", reward_path)
            reward_text = reward_path.read_text(encoding="utf-8").strip()
            reward = float(reward_text)
            reward_details["reward_raw"] = reward_text
        except (FileNotFoundError, OSError, RuntimeError):
            try:
                with _tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                    reward_path = Path(f.name)
                await sandbox.download("/reward.txt", reward_path)
                reward_text = reward_path.read_text(encoding="utf-8").strip()
                reward = float(reward_text)
                reward_details["reward_raw"] = reward_text
            except (FileNotFoundError, OSError, RuntimeError, ValueError):
                reward = 1.0 if result.return_code == 0 else 0.0
                reward_details["reward_source"] = "exit_code_fallback"
        except ValueError:
            reward = 1.0 if result.return_code == 0 else 0.0
            reward_details["reward_source"] = "exit_code_fallback"

        extracted = f"test_exit={result.return_code} reward={reward}"

        return VerifyResult(
            reward=reward,
            extracted_answer=extracted,
            scoring_details=reward_details,
        )

    async def close(self) -> None:
        pass
