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

import asyncio
import atexit
import hashlib
import importlib.util
import io
import json
import logging
import random
import re
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import quote

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import BenchmarkDefinition, ByobEnvironment, scorer
from nemo_evaluator.environments.registry import register
from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)

_DEFAULT_PINCHBENCH_SOURCE_URL = "https://github.com/pinchbench/skill"
_DEFAULT_PINCHBENCH_SOURCE_REF = "47efe9bf5e14ae52dd9764c5e831317442b054a5"  # v2.0.0
_CACHE_DIRS: dict[tuple[str, str], Path] = {}
_CACHE_METADATA: dict[tuple[str, str], dict[str, str]] = {}
_TASK_DATA: dict[tuple[str, str, str], dict[str, Any]] = {}
_UPSTREAM_JUDGE_PROMPT_BUILDERS: dict[Path, Callable[..., str]] = {}
_WORKSPACES: list[Path] = []

_JUDGE_MAX_ATTEMPTS = 5
_JUDGE_BASE_DELAY_S = 4.0
_JUDGE_MAX_DELAY_S = 45.0


def _cleanup_workspaces() -> None:
    for ws in _WORKSPACES:
        try:
            shutil.rmtree(ws, ignore_errors=True)
        except OSError:
            pass


atexit.register(_cleanup_workspaces)


def _cache_key(source_ref: str, source_url: str) -> tuple[str, str]:
    return (source_url.rstrip("/"), source_ref)


def _archive_url(source_url: str, source_ref: str) -> str:
    return f"{source_url.rstrip('/')}/archive/{quote(source_ref, safe='/')}.zip"


def _ensure_cache(
    *,
    source_ref: str = _DEFAULT_PINCHBENCH_SOURCE_REF,
    source_url: str = _DEFAULT_PINCHBENCH_SOURCE_URL,
) -> Path:
    key = _cache_key(source_ref, source_url)
    cached = _CACHE_DIRS.get(key)
    if cached is not None and cached.exists():
        return cached

    archive_url = _archive_url(source_url, source_ref)
    logger.info("Downloading PinchBench tasks from %s", archive_url)
    with urllib.request.urlopen(archive_url, timeout=120) as resp:
        data = resp.read()

    cache = Path(tempfile.mkdtemp(prefix="pinchbench_cache_"))
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(cache)

    extracted = list(cache.iterdir())
    if len(extracted) == 1 and extracted[0].is_dir():
        root = extracted[0]
    else:
        root = cache

    version_file = root / "BENCHMARK_VERSION"
    benchmark_version = version_file.read_text(encoding="utf-8").strip() if version_file.is_file() else ""
    _CACHE_DIRS[key] = root
    _CACHE_METADATA[key] = {
        "source_url": source_url.rstrip("/"),
        "source_ref": source_ref,
        "archive_url": archive_url,
        "benchmark_version": benchmark_version,
    }

    logger.info(
        "PinchBench cached at %s (version=%s ref=%s)",
        root,
        benchmark_version or "unknown",
        source_ref,
    )
    return root


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
        "multi_session": bool(frontmatter.get("multi_session", False)),
        "sessions": frontmatter.get("sessions", []),
        "prerequisites": frontmatter.get("prerequisites", []),
        "workspace_files": frontmatter.get("workspace_files", []),
        "grading_weights": frontmatter.get("grading_weights") or None,
        "prompt": sections.get("Prompt", ""),
        "expected_behavior": sections.get("Expected Behavior", ""),
        "grading_criteria": sections.get("Grading Criteria", ""),
        "grading_code": grading_code,
        "llm_judge_rubric": sections.get("LLM Judge Rubric", ""),
        "source_path": str(path),
    }


def _manifest_task_names(value: Any) -> list[str]:
    names: list[str] = []
    if isinstance(value, str):
        names.append(value)
    elif isinstance(value, list):
        for item in value:
            names.extend(_manifest_task_names(item))
    elif isinstance(value, dict):
        for key in ("tasks", "task_files", "files", "run_first"):
            if key in value:
                names.extend(_manifest_task_names(value[key]))
        for key in ("task", "file", "path", "id"):
            if key in value and isinstance(value[key], str):
                names.append(value[key])
    return names


def _manifest_task_files(tasks_dir: Path) -> list[Path]:
    task_files = {path.name: path for path in sorted(tasks_dir.glob("task_*.md"))}
    if not task_files:
        raise FileNotFoundError(f"No task_*.md files in {tasks_dir}")

    manifest_path = tasks_dir / "manifest.yaml"
    if not manifest_path.is_file():
        return list(task_files.values())

    import yaml

    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse PinchBench manifest %s: %s", manifest_path, exc)
        return list(task_files.values())

    ordered_names: list[str] = []
    if isinstance(manifest, dict):
        ordered_names.extend(_manifest_task_names(manifest.get("run_first", [])))
        categories = manifest.get("categories", {})
        if isinstance(categories, dict):
            for category_value in categories.values():
                ordered_names.extend(_manifest_task_names(category_value))
        elif isinstance(categories, list):
            ordered_names.extend(_manifest_task_names(categories))

    ordered_files: list[Path] = []
    seen: set[str] = set()
    for raw_name in ordered_names:
        name = raw_name if raw_name.endswith(".md") else f"{raw_name}.md"
        path = task_files.get(Path(name).name)
        if path is None:
            logger.warning("PinchBench manifest references missing task file %s", raw_name)
            continue
        if path.name not in seen:
            seen.add(path.name)
            ordered_files.append(path)

    ordered_files.extend(path for name, path in task_files.items() if name not in seen)
    return ordered_files


def _load_all_tasks(
    *,
    source_ref: str = _DEFAULT_PINCHBENCH_SOURCE_REF,
    source_url: str = _DEFAULT_PINCHBENCH_SOURCE_URL,
) -> list[dict[str, Any]]:
    """Download, parse, and return all PinchBench tasks."""
    repo = _ensure_cache(source_ref=source_ref, source_url=source_url)
    source_meta = _CACHE_METADATA.get(
        _cache_key(source_ref, source_url),
        {
            "source_url": source_url.rstrip("/"),
            "source_ref": source_ref,
            "archive_url": _archive_url(source_url, source_ref),
            "benchmark_version": "",
        },
    )
    tasks_dir = repo / "tasks"
    if not tasks_dir.exists():
        raise FileNotFoundError(f"No tasks/ directory in {repo}")

    task_files = _manifest_task_files(tasks_dir)

    rows: list[dict[str, Any]] = []
    source_key = _cache_key(source_ref, source_url)
    for tf in task_files:
        try:
            task = _parse_task(tf)
            task.update(
                {
                    "pinchbench_source_url": source_meta["source_url"],
                    "pinchbench_source_ref": source_meta["source_ref"],
                    "pinchbench_archive_url": source_meta["archive_url"],
                    "pinchbench_benchmark_version": source_meta["benchmark_version"],
                }
            )
            _TASK_DATA[(*source_key, task["id"])] = task
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

    source_ref = row.get("pinchbench_source_ref", _DEFAULT_PINCHBENCH_SOURCE_REF)
    source_url = row.get("pinchbench_source_url", _DEFAULT_PINCHBENCH_SOURCE_URL)
    repo = _ensure_cache(source_ref=source_ref, source_url=source_url)
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
            "timeout_seconds": row.get("timeout_seconds"),
            "multi_session": row.get("multi_session", False),
            "sessions": row.get("sessions", []),
            "prerequisites": row.get("prerequisites", []),
            "pinchbench_source_url": row.get("pinchbench_source_url", _DEFAULT_PINCHBENCH_SOURCE_URL),
            "pinchbench_source_ref": row.get("pinchbench_source_ref", _DEFAULT_PINCHBENCH_SOURCE_REF),
            "pinchbench_archive_url": row.get("pinchbench_archive_url", _archive_url(source_url, source_ref)),
            "pinchbench_benchmark_version": row.get("pinchbench_benchmark_version", ""),
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


def _load_transcript(workspace_path: str, fallback_response: str) -> list[dict[str, Any]]:
    """Load the per-task transcript in upstream envelope shape.

    Prefers ``<workspace>/.nel_transcript.jsonl`` written by the solver
    (see :func:`nemo_evaluator.solvers.openclaw._persist_session_transcript`);
    falls back to a single synthetic assistant message so non-OpenClaw
    solvers (ChatSolver / HarborSolver) still score deterministically.
    """
    transcript: list[dict[str, Any]] = []
    transcript_file = Path(workspace_path) / ".nel_transcript.jsonl" if workspace_path else None
    if transcript_file and transcript_file.exists():
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                transcript.append(json.loads(line))
            except json.JSONDecodeError:
                transcript.append({"raw": line, "parse_error": True})
        if transcript:
            return transcript
    return [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": fallback_response}],
            },
        },
    ]


_WORKSPACE_SKIP_NAMES = frozenset(
    {
        "BOOTSTRAP.md",
        "SOUL.md",
        "USER.md",
        "IDENTITY.md",
        "HEARTBEAT.md",
        "TOOLS.md",
        "AGENTS.md",
    },
)
_WORKSPACE_SKIP_DIRS = frozenset({".git", ".openclaw", "__pycache__", "node_modules", "skills"})
_WORKSPACE_FILE_MAX_CHARS = 3000
_TRANSCRIPT_SUMMARY_MAX_CHARS = 64_000
_WORKSPACE_BLOB_MAX_CHARS = 64_000


def _summarize_transcript(transcript: list[dict[str, Any]]) -> str:
    """Collapse the envelope-shape transcript into a judge-readable summary."""
    parts: list[str] = []
    for event in transcript:
        if not isinstance(event, dict) or event.get("type") != "message":
            continue
        msg = event.get("message") or {}
        role = msg.get("role")
        content = msg.get("content") or []
        if role == "assistant":
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                itype = item.get("type", "")
                if itype in ("toolCall", "tool_use"):
                    args = item.get("arguments") or item.get("input") or item.get("params") or {}
                    truncated: dict[str, Any] = {}
                    for k, v in args.items():
                        if isinstance(v, str) and len(v) > 200:
                            truncated[k] = v[:200] + "...[truncated]"
                        else:
                            truncated[k] = v
                    parts.append(f"Tool: {item.get('name', '')}({json.dumps(truncated)})")
                elif itype == "text":
                    text = (item.get("text") or "").strip()
                    if text:
                        parts.append(f"Assistant: {text[:2000]}")
        elif role in ("toolResult", "tool"):
            if isinstance(content, list):
                rendered = []
                for block in content:
                    if isinstance(block, dict):
                        rendered.append(block.get("text") or str(block))
                    else:
                        rendered.append(str(block))
                body = "\n".join(rendered)
            else:
                body = str(content)
            parts.append(f"Result: {body[:200]}")
        elif role == "user":
            text: str | None = None
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict):
                    text = first.get("text") or first.get("content")
                elif isinstance(first, str):
                    text = first
            elif isinstance(content, str):
                text = content
            if text:
                parts.append(f"User: {text}")
    joined = "\n".join(parts)
    if len(joined) > _TRANSCRIPT_SUMMARY_MAX_CHARS:
        head = _TRANSCRIPT_SUMMARY_MAX_CHARS // 2
        tail = _TRANSCRIPT_SUMMARY_MAX_CHARS - head
        joined = f"{joined[:head]}\n...[transcript truncated]...\n{joined[-tail:]}"
    return joined


def _read_workspace_files_for_judge(workspace_path: str) -> str:
    """Collect agent-authored text file contents for the judge prompt."""
    if not workspace_path:
        return ""
    workspace = Path(workspace_path)
    if not workspace.exists():
        return ""
    blocks: list[str] = []
    total = 0
    for path in sorted(workspace.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(workspace)
        parts = rel.parts
        if any(p.startswith(".") or p in _WORKSPACE_SKIP_DIRS for p in parts):
            continue
        if path.name in _WORKSPACE_SKIP_NAMES:
            continue
        if path.name == ".nel_transcript.jsonl":
            continue
        try:
            content = path.read_text(encoding="utf-8")
            block = f"### File: {rel}\n{content[:_WORKSPACE_FILE_MAX_CHARS]}"
        except (OSError, UnicodeDecodeError):
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            block = f"### File: {rel} (binary or unreadable, {size} bytes)"
        blocks.append(block)
        total += len(block) + 2
        if total >= _WORKSPACE_BLOB_MAX_CHARS:
            blocks.append("...[workspace listing truncated]")
            break
    return "\n\n".join(blocks)


def _load_upstream_judge_prompt_builder(source_root: Path) -> Callable[..., str]:
    root = source_root.resolve()
    cached = _UPSTREAM_JUDGE_PROMPT_BUILDERS.get(root)
    if cached is not None:
        return cached

    module_path = root / "scripts" / "lib_grading.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"PinchBench upstream judge prompt not found at {module_path}")

    module_suffix = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:16]
    module_name = f"_nel_pinchbench_lib_grading_{module_suffix}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import PinchBench upstream judge prompt from {module_path}")

    previous_path = list(sys.path)
    sys.path.insert(0, str(module_path.parent))
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = previous_path

    builder = getattr(module, "_build_judge_prompt", None)
    if not callable(builder):
        raise AttributeError(f"PinchBench upstream module {module_path} does not define _build_judge_prompt")

    _UPSTREAM_JUDGE_PROMPT_BUILDERS[root] = builder
    return builder


def _build_judge_prompt(
    task: dict[str, Any],
    transcript_summary: str,
    rubric: str,
    workspace_content: str,
    *,
    source_root: Path,
) -> str:
    """Render the judge prompt using PinchBench's upstream grading helper."""
    builder = _load_upstream_judge_prompt_builder(source_root)
    upstream_task = SimpleNamespace(
        prompt=task.get("prompt", ""),
        expected_behavior=task.get("expected_behavior", ""),
    )
    return builder(upstream_task, transcript_summary, rubric, workspace_content)


def _resolve_rubric(task: dict[str, Any]) -> str:
    """Pick the most informative scoring rubric available for a task."""
    rubric = task.get("llm_judge_rubric") or ""
    if rubric.strip():
        return rubric
    criteria = task.get("grading_criteria") or ""
    return criteria


_FRACTIONAL_JSON_DECODER = json.JSONDecoder()


def _iter_top_level_json_objects(raw: str) -> list[dict[str, Any]]:
    """Extract every balanced top-level JSON object from ``raw`` text."""
    results: list[dict[str, Any]] = []
    i, n = 0, len(raw)
    while i < n:
        if raw[i] == "{":
            try:
                obj, end = _FRACTIONAL_JSON_DECODER.raw_decode(raw, i)
            except json.JSONDecodeError:
                i += 1
                continue
            if isinstance(obj, dict):
                results.append(obj)
            i = end
        else:
            i += 1
    return results


def _parse_judge_response_fractional(text: str) -> dict[str, Any]:
    """Parse a per-criterion 0.0-1.0 judge envelope into ``{scores, total, notes}``."""
    result: dict[str, Any] = {"scores": {}, "total": None, "notes": "", "raw": text[:2000]}
    if not text:
        return result

    candidates: list[dict[str, Any]] = []
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            candidates.append(obj)
    except json.JSONDecodeError:
        fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fence:
            try:
                obj = json.loads(fence.group(1))
                if isinstance(obj, dict):
                    candidates.append(obj)
            except json.JSONDecodeError:
                pass
        candidates.extend(_iter_top_level_json_objects(text))

    chosen: dict[str, Any] | None = None
    for obj in reversed(candidates):
        if "scores" in obj or "criteria_scores" in obj or "total" in obj or "score" in obj:
            chosen = obj
            break
    if chosen is None and candidates:
        chosen = candidates[-1]

    if chosen is not None:
        raw_scores = chosen.get("scores") or chosen.get("criteria_scores") or {}
        if isinstance(raw_scores, dict):
            for k, v in raw_scores.items():
                if isinstance(v, dict) and "score" in v:
                    v = v.get("score")
                try:
                    result["scores"][str(k)] = float(v)
                except (TypeError, ValueError):
                    continue

        total = chosen.get("total")
        if total is None and "score" in chosen:
            try:
                total = float(chosen["score"])
            except (TypeError, ValueError):
                total = None
        try:
            result["total"] = float(total) if total is not None else None
        except (TypeError, ValueError):
            result["total"] = None

        result["notes"] = str(chosen.get("notes") or chosen.get("justification") or "")
    else:
        m = re.search(r"(?:total|score)\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
        if m:
            try:
                result["total"] = float(m.group(1))
            except ValueError:
                pass

    scores = result["scores"]
    total = result["total"]
    if scores and all(0.0 <= v <= 1.0 for v in scores.values()):
        if total is None or total > 1.0:
            result["total"] = sum(scores.values()) / len(scores)
    if result["total"] is not None:
        result["total"] = max(0.0, min(1.0, float(result["total"])))

    return result


def _make_pinchbench_judge_fn(
    judge_prompt: str,
    automated_score: float | None,
    grading_weights: dict[str, float] | None,
    grading_type: str,
):
    """Build the closure stashed in ``scoring_details["_judge_fn"]``."""

    async def _run(judge_client: Any) -> dict[str, Any]:
        last_exc: Exception | None = None
        model_resp = None
        for attempt in range(_JUDGE_MAX_ATTEMPTS):
            try:
                model_resp = await judge_client.chat(
                    prompt=judge_prompt,
                    system="You are a strict evaluation judge. Always respond with valid JSON.",
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt == _JUDGE_MAX_ATTEMPTS - 1:
                    break
                delay = min(_JUDGE_BASE_DELAY_S * (2**attempt), _JUDGE_MAX_DELAY_S)
                delay *= 0.5 + random.random()
                logger.warning(
                    "PinchBench judge attempt %d/%d failed (%s: %s); sleeping %.1fs",
                    attempt + 1,
                    _JUDGE_MAX_ATTEMPTS,
                    type(exc).__name__,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        if model_resp is None:
            err = f"{type(last_exc).__name__}: {last_exc}" if last_exc else "unknown"
            logger.warning(
                "PinchBench judge call failed after %d attempts (%s): %s",
                _JUDGE_MAX_ATTEMPTS,
                grading_type,
                err,
            )
            judge_info = {"error": err, "total": None, "attempts": _JUDGE_MAX_ATTEMPTS}
            if grading_type == "hybrid" and automated_score is not None:
                return {
                    "reward": float(automated_score),
                    "judge": judge_info,
                    "details": {
                        "hybrid_combine": {
                            "automated": float(automated_score),
                            "judge": None,
                            "combined": float(automated_score),
                            "judge_failed": True,
                        }
                    },
                }
            return {"reward": None, "judge": judge_info}

        parsed = _parse_judge_response_fractional(model_resp.content)
        judge_total = parsed.get("total")
        judge_info = {
            "scores": parsed.get("scores", {}),
            "total": judge_total,
            "notes": parsed.get("notes", ""),
            "judge_model": getattr(model_resp, "model", None),
            "judge_tokens": getattr(model_resp, "total_tokens", None),
            "judge_latency_ms": getattr(model_resp, "latency_ms", None),
        }

        if grading_type == "hybrid" and automated_score is not None:
            w = grading_weights or {"automated": 0.5, "llm_judge": 0.5}
            aw = float(w.get("automated", 0.5))
            jw = float(w.get("llm_judge", 0.5))
            total_w = aw + jw or 1.0
            if judge_total is None:
                reward = float(automated_score)
                combine = {
                    "automated": float(automated_score),
                    "judge": None,
                    "combined": reward,
                }
            else:
                combined = (float(automated_score) * aw + float(judge_total) * jw) / total_w
                reward = combined
                combine = {
                    "automated": float(automated_score),
                    "judge": float(judge_total),
                    "weights": {"automated": aw, "llm_judge": jw},
                    "combined": combined,
                }
            return {
                "reward": reward,
                "judge": judge_info,
                "details": {"hybrid_combine": combine},
            }

        reward = float(judge_total) if judge_total is not None else None
        return {"reward": reward, "judge": judge_info}

    return _run


@scorer
def pinchbench_scorer(sample: ScorerInput) -> dict[str, Any]:
    """Score a PinchBench task.

    Three scoring paths:

    * ``automated`` -- run embedded ``grade()`` only (no judge).
    * ``llm_judge`` -- stash a ``_judge_fn`` closure invoked by the eval
      loop with the judge client.
    * ``hybrid``    -- run ``grade()`` locally and hand its score to the
      closure, which combines via ``grading_weights`` (default 50/50).
    """
    task_id = sample.metadata.get("task_id", "")
    workspace_path = sample.metadata.get("workspace_path", "")

    source_ref = sample.metadata.get("pinchbench_source_ref", _DEFAULT_PINCHBENCH_SOURCE_REF)
    source_url = sample.metadata.get("pinchbench_source_url", _DEFAULT_PINCHBENCH_SOURCE_URL)
    task_info = _TASK_DATA.get((*_cache_key(source_ref, source_url), task_id), {})
    if not task_info:
        task_info = next((task for (*_, cached_task_id), task in _TASK_DATA.items() if cached_task_id == task_id), {})
    grading_type = task_info.get("grading_type") or sample.metadata.get("grading_type", "automated")
    grading_code = task_info.get("grading_code", "")
    grading_weights = task_info.get("grading_weights") or None

    transcript = _load_transcript(workspace_path, sample.response)

    automated_details: dict[str, Any] | None = None
    if grading_code:
        try:
            raw_scores = _run_grade(grading_code, transcript, workspace_path)
        except Exception as exc:
            logger.warning("PinchBench grade() failed for %s: %s", task_id, exc)
            raw_scores = {}
            automated_details = {"automated_score": 0.0, "error": str(exc), "breakdown": {}}
        if automated_details is None:
            if raw_scores:
                avg = sum(raw_scores.values()) / len(raw_scores)
            else:
                avg = 0.0
            automated_details = {
                "automated_score": float(avg),
                "breakdown": {str(k): float(v) for k, v in raw_scores.items()},
            }

    if grading_type in ("llm_judge", "hybrid"):
        rubric = _resolve_rubric(task_info)
        summary = _summarize_transcript(transcript)
        workspace_blob = _read_workspace_files_for_judge(workspace_path)
        source_root = _ensure_cache(source_ref=source_ref, source_url=source_url)
        judge_prompt = _build_judge_prompt(
            task_info,
            summary,
            rubric,
            workspace_blob,
            source_root=source_root,
        )
        automated_score = (
            automated_details["automated_score"] if grading_type == "hybrid" and automated_details is not None else None
        )

        payload: dict[str, Any] = {
            "correct": False,
            "needs_judge": True,
            "_judge_fn": _make_pinchbench_judge_fn(
                judge_prompt=judge_prompt,
                automated_score=automated_score,
                grading_weights=grading_weights,
                grading_type=grading_type,
            ),
            "task_id": task_id,
            "grading_type": grading_type,
            "extracted": sample.response[:500],
        }
        if grading_type == "hybrid" and automated_details is not None:
            payload["automated_breakdown"] = automated_details["breakdown"]
            if "error" in automated_details:
                payload["automated_error"] = automated_details["error"]
        return payload

    if automated_details is None:
        logger.warning(
            "PinchBench task %s is grading_type=automated but has no grading_code",
            task_id,
        )
        return {"correct": 0.0, "error": "no grading_code for automated task"}

    score = automated_details["automated_score"]
    result: dict[str, Any] = {
        "correct": score,
        "reward": score,
        "grading_type": "automated",
        **{f"criterion_{k}": v for k, v in automated_details["breakdown"].items()},
    }
    if "error" in automated_details:
        result["error"] = automated_details["error"]
    return result


def _pinchbench_definition(source_ref: str, source_url: str) -> BenchmarkDefinition:
    def _dataset() -> list[dict[str, Any]]:
        return _load_all_tasks(source_ref=source_ref, source_url=source_url)

    return BenchmarkDefinition(
        name="pinchbench",
        dataset=_dataset,
        prompt="{prompt}",
        target_field="id",
        seed_fn=_seed_fn,
        scorer_fn=pinchbench_scorer,
    )


@register("pinchbench")
class PinchBenchEnvironment(ByobEnvironment):
    def __init__(
        self,
        num_examples: int | None = None,
        source_ref: str = _DEFAULT_PINCHBENCH_SOURCE_REF,
        source_url: str = _DEFAULT_PINCHBENCH_SOURCE_URL,
    ) -> None:
        super().__init__(_pinchbench_definition(source_ref, source_url), num_examples)
