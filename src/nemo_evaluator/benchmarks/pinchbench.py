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
"""PinchBench -- agentic task benchmark.

Tasks are markdown files with YAML frontmatter, a prompt section, grading
criteria, and optionally embedded ``grade()`` functions.  Tasks with automated
grading code are scored directly; the rest use LLM-as-judge via the
``needs_judge`` mechanism with the task's grading criteria as reference.

Solver-independent: works with NatSolver (rich trajectory via
``.nel_transcript.jsonl``), HarborSolver (reads workspace from task JSON),
OpenClawSolver, or even ChatSolver (fallback transcript built from
response text).

Dataset is downloaded once from GitHub as a zip archive.
"""

from __future__ import annotations

import atexit
import io
import json
import logging
import re
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)

_GITHUB_ZIP = "https://github.com/pinchbench/skill/archive/refs/heads/main.zip"
_CACHE_DIR: Path | None = None
_TASK_DATA: dict[str, dict[str, Any]] = {}
_WORKSPACES: list[Path] = []


def _cleanup_workspaces() -> None:
    for ws in _WORKSPACES:
        try:
            shutil.rmtree(ws, ignore_errors=True)
        except OSError:
            pass


atexit.register(_cleanup_workspaces)


def _ensure_cache() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is not None and _CACHE_DIR.exists():
        return _CACHE_DIR

    logger.info("Downloading PinchBench tasks from GitHub...")
    with urllib.request.urlopen(_GITHUB_ZIP, timeout=120) as resp:
        data = resp.read()

    cache = Path(tempfile.mkdtemp(prefix="pinchbench_cache_"))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache)

    extracted = list(cache.iterdir())
    if len(extracted) == 1 and extracted[0].is_dir():
        _CACHE_DIR = extracted[0]
    else:
        _CACHE_DIR = cache

    logger.info("PinchBench cached at %s", _CACHE_DIR)
    return _CACHE_DIR


def _parse_task(path: Path) -> dict[str, Any]:
    """Parse a PinchBench task markdown file into structured fields."""
    content = path.read_text(encoding="utf-8")
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
    if not fm_match:
        raise ValueError(f"No YAML frontmatter in {path}")

    import yaml

    frontmatter = yaml.safe_load(fm_match.group(1))
    body = fm_match.group(2)

    sections: dict[str, str] = {}
    current_header = ""
    current_lines: list[str] = []
    for line in body.splitlines():
        if line.startswith("## "):
            if current_header:
                sections[current_header] = "\n".join(current_lines).strip()
            current_header = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_header:
        sections[current_header] = "\n".join(current_lines).strip()

    grading_code = ""
    automated_checks = sections.get("Automated Checks", "")
    code_match = re.search(r"```python\s*(.*?)\s*```", automated_checks, re.DOTALL)
    if code_match:
        grading_code = code_match.group(1)

    return {
        "id": frontmatter.get("id", path.stem),
        "name": frontmatter.get("name", path.stem),
        "category": frontmatter.get("category", "general"),
        "grading_type": frontmatter.get("grading_type", "automated"),
        "timeout_seconds": frontmatter.get("timeout_seconds", 120),
        "workspace_files": frontmatter.get("workspace_files", []),
        "prompt": sections.get("Prompt", ""),
        "expected_behavior": sections.get("Expected Behavior", ""),
        "grading_criteria": sections.get("Grading Criteria", ""),
        "grading_code": grading_code,
        "llm_judge_rubric": sections.get("LLM Judge Rubric", ""),
        "source_path": str(path),
    }


def _load_all_tasks() -> list[dict[str, Any]]:
    """Download, parse, and return all PinchBench tasks."""
    repo = _ensure_cache()
    tasks_dir = repo / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"No tasks/ directory in {repo}")

    task_files = sorted(tasks_dir.glob("task_*.md"))
    if not task_files:
        raise FileNotFoundError(f"No task_*.md files in {tasks_dir}")

    rows: list[dict[str, Any]] = []
    for tf in task_files:
        try:
            task = _parse_task(tf)
            _TASK_DATA[task["id"]] = task
            rows.append(task)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", tf, exc)

    automated = sum(1 for r in rows if r.get("grading_code"))
    logger.info(
        "PinchBench: loaded %d tasks (%d automated, %d judge-scored)", len(rows), automated, len(rows) - automated
    )
    return rows


def _seed_fn(row: dict[str, Any], idx: int) -> SeedResult:
    """Create a SeedResult with a per-task workspace directory."""
    workspace = Path(tempfile.mkdtemp(prefix=f"pinch_{row['id']}_"))
    _WORKSPACES.append(workspace)

    repo = _ensure_cache()
    assets_dir = repo / "assets"
    for file_spec in row.get("workspace_files", []):
        if "content" in file_spec:
            dest = workspace / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"], encoding="utf-8")
        elif "source" in file_spec:
            src = assets_dir / file_spec["source"]
            dest = workspace / file_spec["dest"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                shutil.copy2(src, dest)
            else:
                logger.warning("Asset not found: %s", src)

    expected = ""
    if not row.get("grading_code"):
        parts = []
        if row.get("expected_behavior"):
            parts.append(f"Expected behavior:\n{row['expected_behavior']}")
        if row.get("grading_criteria"):
            parts.append(f"Grading criteria:\n{row['grading_criteria']}")
        expected = "\n\n".join(parts)

    return SeedResult(
        prompt=row["prompt"],
        expected_answer=expected,
        metadata={
            "source": "pinchbench",
            "task_id": row["id"],
            "task_name": row["name"],
            "category": row["category"],
            "workspace_path": str(workspace),
            "grading_type": row["grading_type"],
        },
    )


def _run_grade(grading_code: str, transcript: list, workspace_path: str) -> dict[str, float]:
    """Execute a PinchBench grade() function in an isolated namespace."""
    namespace: dict[str, Any] = {}
    exec(grading_code, namespace)  # noqa: S102
    grade_fn = namespace.get("grade")
    if grade_fn is None:
        return {"error": 0.0}
    return grade_fn(transcript, workspace_path)


@benchmark(
    name="pinchbench",
    dataset=_load_all_tasks,
    prompt="{prompt}",
    target_field="id",
    seed_fn=_seed_fn,
)
@scorer
def pinchbench_scorer(sample: ScorerInput) -> dict[str, Any]:
    """Score a PinchBench task.  Solver-independent:

    1. If ``.nel_transcript.jsonl`` exists in workspace (written by NatSolver),
       use the rich trajectory.
    2. Otherwise, build a minimal transcript from the solver's text response.
       This enables ChatSolver, HarborSolver, etc. to be scored -- the grade
       function may not find tool calls but can still check workspace files.
    """
    task_id = sample.metadata.get("task_id", "")
    workspace_path = sample.metadata.get("workspace_path", "")

    task_info = _TASK_DATA.get(task_id, {})
    grading_code = task_info.get("grading_code", "")

    if not grading_code:
        from nemo_evaluator.scoring.judge import needs_judge

        return needs_judge(sample)

    transcript: list[dict[str, Any]] = []
    transcript_file = Path(workspace_path) / ".nel_transcript.jsonl" if workspace_path else None

    if transcript_file and transcript_file.exists():
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    transcript.append(json.loads(line))
                except json.JSONDecodeError:
                    transcript.append({"raw": line, "parse_error": True})
    else:
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample.response}],
                },
            }
        ]

    try:
        scores = _run_grade(grading_code, transcript, workspace_path)
    except Exception as exc:
        logger.warning("PinchBench grade() failed for %s: %s", task_id, exc)
        return {"correct": 0.0, "error": str(exc)}

    if not scores:
        return {"correct": 0.0}

    avg = sum(scores.values()) / len(scores)
    return {
        "correct": avg,
        "reward": avg,
        **{f"criterion_{k}": v for k, v in scores.items()},
    }
