"""SWE-bench shared logic for Verified and Multilingual variants."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.sandbox.base import ImageBuildRequest, ImageSpec, SandboxSpec
from nemo_evaluator.scoring.types import ScorerInput

logger = logging.getLogger(__name__)

_MIN_RUST_VERSION = "1.88"


def _patch_swebench_rust_versions(min_version: str = _MIN_RUST_VERSION) -> None:
    """Bump swebench Rust docker_specs to *min_version* so that freshly-resolved
    crates (e.g. ``time >= 0.3.47``) can compile.  Swebench pins older Rust
    toolchains that are below the MSRV of current transitive dependencies.
    """
    try:
        from swebench.harness.constants.rust import MAP_REPO_VERSION_TO_SPECS_RUST
    except ImportError:
        return
    min_parts = tuple(int(x) for x in min_version.split("."))
    patched = 0
    for _repo, version_specs in MAP_REPO_VERSION_TO_SPECS_RUST.items():
        for _ver, spec in version_specs.items():
            ds = spec.get("docker_specs", {})
            cur = ds.get("rust_version")
            if cur is None:
                continue
            cur_parts = tuple(int(x) for x in cur.split("."))
            if cur_parts < min_parts:
                ds["rust_version"] = min_version
                patched += 1
    if patched:
        logger.info("Patched %d swebench Rust specs to rust_version >= %s", patched, min_version)


_MICROPYTHON_EXTRA_PKGS = ["libffi-dev", "pkg-config"]


def _patch_swebench_c_deps() -> None:
    """Prepend ``apt-get install libffi-dev pkg-config`` to micropython
    ``pre_install`` so they run inside ``setup_repo.sh`` (instance image build).

    The swebench C language has no env Dockerfile, so ``apt-pkgs`` in specs
    generates a ``setup_env.sh`` that is never executed.  ``pre_install`` is
    the correct hook for instance-level system deps.
    """
    try:
        from swebench.harness.constants.c import SPECS_MICROPYTHON
    except ImportError:
        return
    apt_cmds = ["apt-get update", f"apt-get install -y {' '.join(_MICROPYTHON_EXTRA_PKGS)}"]
    patched = 0
    for _ver, spec in SPECS_MICROPYTHON.items():
        existing = spec.get("pre_install", [])
        if not any("libffi-dev" in cmd for cmd in existing):
            spec["pre_install"] = apt_cmds + existing
            patched += 1
    if patched:
        logger.info("Patched %d micropython specs with pre_install apt deps", patched)


SWEBENCH_IMAGE_TEMPLATE = "swebench/sweb.eval.x86_64.{instance_id}:latest"

_LOCAL_BUILD_TEMPLATE = "sweb.eval.x86_64.{instance_id}:latest"

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


def swebench_codebuild_buildspec(
    specs: list[ImageSpec],
    ecr_repo: str,
    ecr_region: str | None,
    dockerhub_secret_arn: str | None = None,
    *,
    dataset_name: str = "SWE-bench/SWE-bench",
) -> str:
    """Generate a CodeBuild buildspec that builds swebench images and pushes to ECR.

    Used as a fallback when the local Docker daemon is unavailable (e.g. SLURM).
    CodeBuild runs in privileged mode so the swebench harness can build Docker
    images remotely.
    """
    instance_ids = [s.source["instance_id"] for s in specs]
    ids_json = json.dumps(instance_ids)
    ecr_registry = ecr_repo.split("/")[0]

    region_flag = ecr_region or "$AWS_DEFAULT_REGION"
    pre_build_cmds = [
        f"aws ecr get-login-password --region {region_flag}"
        f" | docker login --username AWS --password-stdin {ecr_registry}",
    ]
    if dockerhub_secret_arn:
        pre_build_cmds.append(
            f"DOCKERHUB_CREDS=$(aws secretsmanager get-secret-value"
            f" --secret-id {dockerhub_secret_arn}"
            f" --query SecretString --output text --region $AWS_DEFAULT_REGION)"
            f' && DH_USER=$(echo "$DOCKERHUB_CREDS" | python3 -c'
            """ "import sys,json;print(json.load(sys.stdin)['username'])")"""
            f' && if [ -n "$DH_USER" ]; then echo "$DOCKERHUB_CREDS" | python3 -c'
            """ "import sys,json;print(json.load(sys.stdin)['password'])" """
            f'| docker login -u "$DH_USER" --password-stdin; fi'
            f' || echo "Docker Hub login failed — continuing without auth"'
        )

    pre_yaml = "\n".join(f"      - {c}" for c in pre_build_cmds)

    # The build script is a self-contained Python program that:
    # 1. Builds env + instance images via the swebench harness
    # 2. Tags each with the _sanitize_id ECR convention
    # 3. Pushes to ECR
    build_script = f"""\
import json, re, subprocess, sys

instance_ids = json.loads('''{ids_json}''')
ecr_repo = '{ecr_repo}'
dataset_name = '{dataset_name}'

def sanitize(s, max_len=100):
    return re.sub(r'[^a-zA-Z0-9-]+', '-', s).strip('-')[:max_len] or 'task'

def image_name(iid):
    safe = iid.replace('/', '__').replace(':', '_')
    return f'swebench/sweb.eval.x86_64.{{safe}}:latest'

print(f'Building {{len(instance_ids)}} SWE-bench images from {{dataset_name}}', flush=True)

try:
    import docker
    from swebench.harness.docker_build import build_env_images, build_instance_images
    from swebench.harness.utils import load_swebench_dataset
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'swebench', 'docker', 'datasets'])
    import docker
    from swebench.harness.docker_build import build_env_images, build_instance_images
    from swebench.harness.utils import load_swebench_dataset

client = docker.from_env()

# Patch swebench Rust specs so cargo can handle crates with high MSRV (e.g. time>=0.3.47 needs 1.88)
try:
    from swebench.harness.constants.rust import MAP_REPO_VERSION_TO_SPECS_RUST
    _min = (1, 88)
    _patched = 0
    for _repo, _vspecs in MAP_REPO_VERSION_TO_SPECS_RUST.items():
        for _v, _s in _vspecs.items():
            _ds = _s.get('docker_specs', {{}})
            _cv = _ds.get('rust_version')
            if _cv and tuple(int(x) for x in _cv.split('.')) < _min:
                _ds['rust_version'] = '1.88'
                _patched += 1
    if _patched:
        print(f'Patched {{_patched}} swebench Rust specs to rust_version >= 1.88', flush=True)
except Exception:
    pass

# Patch swebench micropython specs: prepend apt-get install to pre_install
# so libffi-dev and pkg-config are available during instance image build.
try:
    from swebench.harness.constants.c import SPECS_MICROPYTHON
    _apt_cmds = ['apt-get update', 'apt-get install -y libffi-dev pkg-config']
    for _v, _s in SPECS_MICROPYTHON.items():
        _existing = _s.get('pre_install', [])
        if not any('libffi-dev' in c for c in _existing):
            _s['pre_install'] = _apt_cmds + _existing
except Exception:
    pass

dataset = load_swebench_dataset(name=dataset_name, instance_ids=instance_ids)

print('Building env images...', flush=True)
build_env_images(client=client, dataset=dataset, max_workers=4,
                 instance_image_tag='latest', env_image_tag='latest')

print('Building instance images...', flush=True)
build_instance_images(client=client, dataset=dataset, max_workers=4,
                      tag='latest', env_image_tag='latest')

# Re-tag from swebench local names to hub-style names
for iid in instance_ids:
    safe = iid.replace('/', '__').replace(':', '_')
    local = f'sweb.eval.x86_64.{{safe}}:latest'
    hub = f'swebench/sweb.eval.x86_64.{{safe}}:latest'
    subprocess.run(['docker', 'tag', local, hub], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Tag and push to ECR
failed = 0
for i, iid in enumerate(instance_ids, 1):
    local = image_name(iid)
    tag = sanitize(local)
    ecr_url = f'{{ecr_repo}}:{{tag}}'
    print(f'  [{{i}}/{{len(instance_ids)}}] {{iid}} -> {{tag}}', flush=True)
    subprocess.run(['docker', 'tag', local, ecr_url], check=True)
    r = subprocess.run(['docker', 'push', ecr_url], capture_output=True, text=True)
    if r.returncode != 0:
        print(f'    FAILED: {{r.stderr.strip()[:200]}}')
        failed += 1

if failed:
    print(f'WARNING: {{failed}}/{{len(instance_ids)}} pushes failed')
    sys.exit(1)
print(f'Done: {{len(instance_ids)}} images pushed to ECR')
"""

    # Base64-encode the script to avoid YAML/shell escaping issues
    import base64

    encoded_script = base64.b64encode(build_script.encode()).decode()

    return (
        "version: 0.2\n"
        "phases:\n"
        "  install:\n"
        "    commands:\n"
        "      - pip3 install swebench docker datasets\n"
        "  pre_build:\n"
        "    commands:\n"
        f"{pre_yaml}\n"
        "  build:\n"
        "    commands:\n"
        f"      - echo '{encoded_script}' | base64 -d | python3\n"
    )


def swebench_image_build_request(
    rows: list[dict[str, Any]],
    *,
    dataset_name: str = "SWE-bench/SWE-bench",
) -> ImageBuildRequest:
    instance_ids = sorted({r["instance_id"] for r in rows if "instance_id" in r})
    specs = [
        ImageSpec(
            image=_instance_id_to_image(iid),
            source={"instance_id": iid},
        )
        for iid in instance_ids
    ]

    def _docker_build(specs: list[ImageSpec]) -> None:
        _build_swebench_docker(specs, dataset_name=dataset_name)

    def _codebuild_buildspec(
        specs: list[ImageSpec],
        ecr_repo: str,
        ecr_region: str,
        dockerhub_secret_arn: str | None = None,
    ) -> str:
        return swebench_codebuild_buildspec(
            specs,
            ecr_repo,
            ecr_region,
            dockerhub_secret_arn,
            dataset_name=dataset_name,
        )

    return ImageBuildRequest(
        specs=specs,
        docker_build_fn=_docker_build,
        codebuild_buildspec_fn=_codebuild_buildspec,
    )


def _build_swebench_docker(
    specs: list[ImageSpec],
    *,
    dataset_name: str = "SWE-bench/SWE-bench",
) -> None:
    import docker as docker_mod

    client = docker_mod.from_env()
    client.ping()

    missing_ids = [s.source["instance_id"] for s in specs]
    if not missing_ids:
        return

    logger.info("Building %d SWE-bench Docker images from %s", len(missing_ids), dataset_name)

    try:
        from swebench.harness.docker_build import build_env_images, build_instance_images
        from swebench.harness.utils import load_swebench_dataset
    except ImportError:
        raise ImportError(
            "SWE-bench image building requires the 'swebench' and 'docker' "
            "packages.  Install with: pip install swebench docker"
        )

    _patch_swebench_rust_versions()
    _patch_swebench_c_deps()

    dataset = load_swebench_dataset(name=dataset_name, instance_ids=missing_ids)
    # Explicit "latest" tags work around a swebench positional-arg bug:
    # get_test_specs_from_dataset passes instance_image_tag where
    # make_test_spec expects base_image_tag; None triggers an assertion.
    build_env_images(
        client=client,
        dataset=dataset,
        max_workers=4,
        instance_image_tag="latest",
        env_image_tag="latest",
    )
    build_instance_images(
        client=client,
        dataset=dataset,
        max_workers=4,
        tag="latest",
        env_image_tag="latest",
    )

    for iid in missing_ids:
        local_name = _instance_id_to_image(iid, _LOCAL_BUILD_TEMPLATE)
        hub_name = _instance_id_to_image(iid, SWEBENCH_IMAGE_TEMPLATE)
        if local_name != hub_name:
            subprocess.run(
                ["docker", "tag", local_name, hub_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


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
        "FAIL_TO_PASS": _ensure_str(row.get("FAIL_TO_PASS", "")),
        "PASS_TO_PASS": _ensure_str(row.get("PASS_TO_PASS", "")),
    }


def _ensure_str(val: Any) -> str:
    """Coerce list-typed fields (common in HF datasets) to JSON strings."""
    if isinstance(val, list):
        return json.dumps(val)
    return val if isinstance(val, str) else str(val)


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
            "FAIL_TO_PASS": _ensure_str(row.get("FAIL_TO_PASS", "")),
            "PASS_TO_PASS": _ensure_str(row.get("PASS_TO_PASS", "")),
        },
        sandbox_spec=agent_spec,
        verify_sandbox_spec=verify_spec,
        capture_cmd=_CAPTURE_CMD,
        apply_cmd=_APPLY_CMD,
    )


# ---------------------------------------------------------------------------
# Verification runner (executed inside the sandbox via ``uv run``)
# ---------------------------------------------------------------------------

_RUNNER_PY = """\
# /// script
# requires-python = ">=3.11"
# dependencies = ["swebench==4.0.3", "datasets==2.16.1", "fastcore<1.11"]
# ///

import json
import subprocess
import sys

from swebench.harness.constants import (
    EvalType,
    FAIL_ONLY_REPOS,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    PASS_TO_PASS,
    ResolvedStatus,
)
from swebench.harness.grading import (
    get_eval_tests_report,
    get_logs_eval,
    get_resolution_status,
)
from swebench.harness.test_spec.test_spec import make_test_spec

EVAL_SCRIPT = "/tmp/swebench_eval.sh"
EVAL_LOG = "/tmp/swebench_eval.log"
RESULT_FILE = "/tmp/swebench_result.json"

with open("/tmp/swebench_config.json") as f:
    instance = json.load(f)

test_spec = make_test_spec(instance)

with open(EVAL_SCRIPT, "w") as f:
    f.write(test_spec.eval_script)

try:
    with open(EVAL_LOG, "w") as log_f:
        proc = subprocess.run(
            ["bash", EVAL_SCRIPT],
            stdout=log_f,
            stderr=subprocess.STDOUT,
            timeout=1500,
        )
    eval_exit_code = proc.returncode
except subprocess.TimeoutExpired:
    eval_exit_code = -1

result = {
    "instance_id": instance.get("instance_id", ""),
    "patch_successfully_applied": False,
    "resolved": False,
    "tests_status": {},
    "eval_exit_code": eval_exit_code,
}

eval_status_map, found = get_logs_eval(test_spec, EVAL_LOG)

if found:
    result["patch_successfully_applied"] = True
    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    eval_type = (
        EvalType.FAIL_ONLY
        if test_spec.repo in FAIL_ONLY_REPOS
        else EvalType.PASS_AND_FAIL
    )
    report = get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
    if get_resolution_status(report) == ResolvedStatus.FULL.value:
        result["resolved"] = True
    result["tests_status"] = report

with open(RESULT_FILE, "w") as f:
    json.dump(result, f)

sys.exit(0)
"""

_VERIFY_WRAPPER = """\
#!/bin/bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &> /dev/null; then
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget &> /dev/null; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo '{"error":"no_curl_or_wget","resolved":false}' > /tmp/swebench_result.json
        exit 0
    fi
fi

uv run /tmp/swebench_runner.py
"""


async def swebench_score(sample: ScorerInput) -> dict[str, Any]:
    """Run swebench evaluation inside the verify sandbox.

    Uploads a config JSON and a PEP 723 runner script, then executes the
    runner via ``uv run``.  The runner uses ``swebench.harness`` (auto-
    installed by ``uv``) to generate the correct eval script for any
    language, run the tests, parse output with the repo-specific log
    parser, and grade pass/fail.  Results are written to a JSON file
    that we read back.
    """
    sandbox = sample.sandbox
    if sandbox is None:
        return {"correct": 0.0, "error": "no_sandbox"}

    meta = sample.metadata
    config = {
        "instance_id": meta.get("instance_id", ""),
        "repo": meta.get("repo", ""),
        "version": meta.get("version", ""),
        "base_commit": meta.get("base_commit", ""),
        "test_patch": meta.get("test_patch", ""),
        "FAIL_TO_PASS": _normalise_test_list(meta.get("FAIL_TO_PASS", "")),
        "PASS_TO_PASS": _normalise_test_list(meta.get("PASS_TO_PASS", "")),
    }

    config_tmp = Path(tempfile.mktemp(suffix=".json"))
    runner_tmp = Path(tempfile.mktemp(suffix=".py"))
    try:
        config_tmp.write_text(json.dumps(config))
        runner_tmp.write_text(_RUNNER_PY)
        await sandbox.upload(config_tmp, "/tmp/swebench_config.json")
        await sandbox.upload(runner_tmp, "/tmp/swebench_runner.py")
    finally:
        config_tmp.unlink(missing_ok=True)
        runner_tmp.unlink(missing_ok=True)

    exec_result = await sandbox.exec(_VERIFY_WRAPPER, timeout_sec=1800)

    cat = await sandbox.exec("cat /tmp/swebench_result.json", timeout_sec=30)
    try:
        report = json.loads(cat.stdout)
    except (json.JSONDecodeError, TypeError):
        logger.error(
            "swebench_score: result JSON missing or invalid for %s\n"
            "wrapper stdout (last 2000): %s\nwrapper stderr (last 2000): %s",
            meta.get("instance_id", "?"),
            (exec_result.stdout or "")[-2000:],
            (exec_result.stderr or "")[-2000:],
        )
        return {
            "correct": 0.0,
            "error": "result_parse_error",
            "test_output_tail": (exec_result.stdout or "")[-2000:],
        }

    resolved = report.get("resolved", False)
    return {
        "correct": 1.0 if resolved else 0.0,
        "resolved": resolved,
        "patch_applied": report.get("patch_successfully_applied", False),
        "tests_status": report.get("tests_status", {}),
        "eval_exit_code": report.get("eval_exit_code"),
        "test_output_tail": (exec_result.stdout or "")[-2000:],
    }


def _normalise_test_list(val: Any) -> list:
    """Ensure FAIL_TO_PASS / PASS_TO_PASS is a proper Python list.

    Values arrive as Python list, JSON string, empty string, or None.
    ``make_test_spec._from_json_or_obj`` handles both list and JSON-string
    inputs, but normalising here keeps the config.json clean.
    """
    if isinstance(val, list):
        return val
    if not isinstance(val, str) or not val.strip():
        return []
    val = val.strip()
    if val.startswith("["):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            pass
    return [val]
