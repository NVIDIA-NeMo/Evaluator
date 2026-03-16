"""SWE-bench shared logic for Verified and Multilingual variants.

Provides ``swebench_seed_fn``, ``swebench_score``, and ``swebench_prepare_row``
that both ``swebench_verified.py`` and ``swebench_multilingual.py`` import.

Architecture:
    - ``seed_fn`` returns a ``SeedResult`` with *both* ``sandbox_spec`` (agent
      container) and ``verify_sandbox_spec`` (fresh verification container),
      plus a ``capture_cmd`` / ``apply_cmd`` pair.  This triggers the
      ``StatelessSandbox`` lifecycle in the eval loop.
    - The scorer is **pure test execution** — the patch is already applied by
      the lifecycle's ``apply_cmd`` before the scorer runs.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.sandbox.base import SandboxSpec
from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)

SWEBENCH_IMAGE_TEMPLATE = "swebench/sweb.eval.x86_64.{instance_id}:latest"

SWEBENCH_PROMPT = (
    "You are an expert software engineer. You will be given a GitHub issue "
    "and the repository where it occurs. Your task is to produce a minimal "
    "git patch (unified diff format) that resolves the issue.\n\n"
    "## Repository\n{repo} (version {version})\n\n"
    "## Issue\n{problem_statement}\n\n"
    "Please respond with ONLY the patch in unified diff format, enclosed in "
    "a code block:\n```diff\n<your patch here>\n```"
)

_CAPTURE_CMD = "cd /testbed && git diff HEAD > /output/patch.diff 2>/dev/null || true"

_APPLY_CMD = (
    "cd /testbed && "
    "if [ -s /input/patch.diff ]; then git apply /input/patch.diff 2>/dev/null || "
    "git apply --reject /input/patch.diff 2>/dev/null || true; "
    "elif [ -f /input/response.txt ]; then git apply /input/response.txt 2>/dev/null || "
    "git apply --reject /input/response.txt 2>/dev/null || true; fi"
)


def _instance_id_to_image(instance_id: str, template: str = SWEBENCH_IMAGE_TEMPLATE) -> str:
    safe_id = instance_id.replace("/", "__").replace(":", "_")
    return template.format(instance_id=safe_id)


def swebench_prepare_row(row: dict[str, Any], idx: int, rng: Any) -> dict[str, Any]:
    """Normalize HF dataset rows for SWE-bench."""
    return {
        "instance_id": row["instance_id"],
        "repo": row.get("repo", ""),
        "version": row.get("version", ""),
        "base_commit": row.get("base_commit", ""),
        "problem_statement": row.get("problem_statement", ""),
        "hints_text": row.get("hints_text", ""),
        "test_patch": row.get("test_patch", ""),
        "test_cmd": row.get("test_cmd", _default_test_cmd(row)),
        "FAIL_TO_PASS": row.get("FAIL_TO_PASS", ""),
        "PASS_TO_PASS": row.get("PASS_TO_PASS", ""),
    }


def _default_test_cmd(row: dict[str, Any]) -> str:
    """Derive a test command when not explicitly provided."""
    repo = row.get("repo", "")
    if "django" in repo:
        return "python -m pytest --no-header -rN -p no:cacheprovider"
    if "scikit-learn" in repo or "sklearn" in repo:
        return "python -m pytest --no-header -rN -p no:cacheprovider"
    return "python -m pytest --no-header -rN -p no:cacheprovider --tb=short"


def swebench_seed_fn(
    row: dict[str, Any],
    idx: int,
    *,
    image_template: str = SWEBENCH_IMAGE_TEMPLATE,
) -> SeedResult:
    """Create a SeedResult with dual sandbox specs for two-container evaluation."""
    instance_id = row["instance_id"]
    image = _instance_id_to_image(instance_id, image_template)

    agent_spec = SandboxSpec(image=image, workdir="/testbed")
    verify_spec = SandboxSpec(image=image, workdir="/testbed")

    prompt_text = SWEBENCH_PROMPT.format(
        repo=row.get("repo", "unknown"),
        version=row.get("version", "unknown"),
        problem_statement=row.get("problem_statement", ""),
    )
    if row.get("hints_text"):
        prompt_text += f"\n\n## Hints\n{row['hints_text']}"

    return SeedResult(
        prompt=prompt_text,
        expected_answer=instance_id,
        metadata={
            "source": "swebench",
            "instance_id": instance_id,
            "repo": row.get("repo", ""),
            "version": row.get("version", ""),
            "base_commit": row.get("base_commit", ""),
            "test_patch": row.get("test_patch", ""),
            "test_cmd": row.get("test_cmd", ""),
            "FAIL_TO_PASS": row.get("FAIL_TO_PASS", ""),
            "PASS_TO_PASS": row.get("PASS_TO_PASS", ""),
        },
        sandbox_spec=agent_spec,
        verify_sandbox_spec=verify_spec,
        capture_cmd=_CAPTURE_CMD,
        apply_cmd=_APPLY_CMD,
    )


async def swebench_score(sample: ScorerInput) -> dict[str, Any]:
    """Score a SWE-bench problem.  Pure test execution — patch already applied.

    Steps:
        1. Apply the test_patch (adds/modifies test files).
        2. Run the test command.
        3. Check that FAIL_TO_PASS tests now pass.
    """
    sandbox = sample.sandbox
    if sandbox is None:
        return {"correct": 0.0, "error": "no_sandbox"}

    test_patch = sample.metadata.get("test_patch", "")
    test_cmd = sample.metadata.get("test_cmd", "")
    fail_to_pass_raw = sample.metadata.get("FAIL_TO_PASS", "")
    pass_to_pass_raw = sample.metadata.get("PASS_TO_PASS", "")

    fail_to_pass = _parse_test_list(fail_to_pass_raw)
    pass_to_pass = _parse_test_list(pass_to_pass_raw)

    if test_patch:
        apply_result = await sandbox.exec(
            f"cd /testbed && echo {_shell_quote(test_patch)} | git apply --allow-empty -",
            timeout_sec=60,
        )
        if apply_result.return_code != 0:
            logger.warning("test_patch apply failed: %s", apply_result.stderr[:300])

    if not test_cmd:
        return {"correct": 0.0, "error": "no_test_cmd"}

    result = await sandbox.exec(f"cd /testbed && {test_cmd}", timeout_sec=300)

    output = result.stdout + result.stderr
    passed_tests, failed_tests = _parse_pytest_output(output)

    f2p_resolved = all(t in passed_tests for t in fail_to_pass) if fail_to_pass else result.return_code == 0
    p2p_intact = all(t not in failed_tests for t in pass_to_pass)
    correct = 1.0 if (f2p_resolved and p2p_intact) else 0.0

    return {
        "correct": correct,
        "fail_to_pass_resolved": f2p_resolved,
        "pass_to_pass_intact": p2p_intact,
        "tests_passed": passed_tests[:20],
        "tests_failed": failed_tests[:20],
        "test_exit_code": result.return_code,
        "test_output_tail": output[-2000:] if output else "",
    }


def _parse_test_list(raw: str) -> list[str]:
    """Parse a JSON-encoded list or comma-separated string of test names."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("["):
        import json
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return [t.strip() for t in raw.split(",") if t.strip()]


def _parse_pytest_output(output: str) -> tuple[list[str], list[str]]:
    """Extract passed and failed test names from pytest output."""
    passed: list[str] = []
    failed: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("PASSED") or " PASSED" in line:
            name = re.split(r"\s+PASSED", line)[0].strip()
            if name:
                passed.append(name)
        elif line.startswith("FAILED") or " FAILED" in line:
            name = re.split(r"\s+FAILED", line)[0].strip()
            if name:
                failed.append(name)
    return passed, failed


def _shell_quote(s: str) -> str:
    """Single-quote a string for shell use."""
    return "'" + s.replace("'", "'\\''") + "'"
