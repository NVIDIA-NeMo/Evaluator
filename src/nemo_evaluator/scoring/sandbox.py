"""Docker-sandboxed code execution scorer."""
from __future__ import annotations

import re

from nemo_evaluator.scoring.types import ScorerInput


def code_sandbox(sample: ScorerInput) -> dict:
    """Run code in a hardened Docker sandbox. Pipes code via stdin."""
    import subprocess

    prompt_code = sample.metadata.get("_prompt", "")
    test_code = sample.metadata.get("_test", "")
    entry_point = sample.metadata.get("entry_point", "solution")

    m = re.search(r"```(?:python)?\s*\n((?:\n|.)+?)```", sample.response, re.DOTALL)
    completion = m.group(1) if m else sample.response

    code = f"{prompt_code}{completion}\n{test_code}\ncheck({entry_point})"
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm", "-i",
                "--network", "none",
                "--memory", "256m",
                "--cpus", "1",
                "--pids-limit", "64",
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "--security-opt", "no-new-privileges",
                "python:3.12-slim", "python", "-",
            ],
            input=code, capture_output=True, text=True, timeout=30,
        )
        passed = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        passed = False

    return {"correct": passed, "extracted": completion[:500]}
