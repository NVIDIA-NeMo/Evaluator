"""Manual ECS smoke test for Terminal-Bench v1 `install-windows-xp`.

This script starts one ECS Fargate sandbox for the mapped task, waits for
SSH/exec health, runs a few in-container diagnostics, then cleans up. It is
intended for manual review/debugging of ECS sandbox image behavior, not for CI.

Prerequisites:

* AWS credentials with access to the ECS sandbox account. The script can load
  `export KEY=value` lines from `--env-file`; if that file is absent it uses
  the current shell environment.
* A config whose first benchmark is `terminal-bench-hard` or
  `terminal-bench-hard-aa-split` and whose sandbox is `ecs_fargate`.
* `nemo-evaluator[harbor]` installed in the environment used to run Python.

Common commands:

    # Dry-run only: map/load the dataset, apply any dataset patch, and print
    # the environment directory plus expected ECR image tag. This does not
    # start ECS.
    python examples/dev/debug_install_windows_xp_sandbox.py \
      --config /path/to/terminal-bench-hard-aa-split_api.yaml \
      --datasets-dir /tmp/nel-xp-cache \
      --dry-run

    # Real ECS smoke test. This builds or reuses the ECR image for the mapped
    # environment, starts one Fargate task, waits for the exec server, verifies
    # the XP wrapper/services, then stops the task and deregisters the task def.
    python examples/dev/debug_install_windows_xp_sandbox.py \
      --config /path/to/terminal-bench-hard-aa-split_api.yaml \
      --datasets-dir /tmp/nel-xp-cache

    # Use a repo-local .env file for AWS/API credentials instead of relying on
    # the current shell environment.
    python examples/dev/debug_install_windows_xp_sandbox.py \
      --config /path/to/terminal-bench-hard-aa-split_api.yaml \
      --env-file .env \
      --datasets-dir /tmp/nel-xp-cache

Important arguments:

* `--config` supplies the ECS sandbox settings: region, ECR repository,
  resources, SSM project, log prefix, and sidecar settings. The default points
  at the repo example config, but review configs can be passed explicitly.
* `--datasets-dir` controls the Harbor dataset cache. Use a fresh temp
  directory to force a fresh Terminal-Bench mapping; reuse the same directory
  to test behavior on an existing `.tbv1_mapped` cache.
* `--nel-next-root` points imports at this checkout's `src/`; override it only
  when running the script from outside this repository.
* `--diagnostic-command` can be repeated to replace the default diagnostics.
  Defaults verify `/nel-entrypoint.sh`, key supervisor-managed services, and
  that `http://127.0.0.1` responds from inside the sandbox.

To prove this specific fix, use a source-level A/B rather than modifying this
script:

1. Restore `src/nemo_evaluator/benchmarks/terminal_bench_v1.py` from
   `origin/main` and run the script with a fresh `--datasets-dir`. Expected:
   the mapped Dockerfile keeps upstream's `supervisord` ENTRYPOINT, the ECR tag
   matches the broken content hash, and ECS startup fails with the main
   container exiting before SSH/exec is usable.
2. Restore the MR version of `terminal_bench_v1.py` and rerun the script
   against the same `--datasets-dir`. Expected: `_ensure_dataset()` patches the
   existing cache, the ECR tag changes to the wrapper-patched content hash, ECS
   reaches RUNNING, exec health is ready, and the default diagnostics pass.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "examples/configs/08e_terminalbench_hard_oracle_fargate.yaml"
DEFAULT_ENV_FILE = REPO_ROOT / ".env"
DEFAULT_DIAGNOSTICS = [
    "test -x /nel-entrypoint.sh && echo nel-entrypoint-present",
    'ps -ef | grep -E "supervisord|nginx|websockify|Xvnc|pulseaudio|qemu" | grep -v grep',
    "wget -qO- -T 5 http://127.0.0.1 | head -c 80",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nel-next-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--datasets-dir", type=Path)
    parser.add_argument("--task-id", default="install-windows-xp")
    parser.add_argument("--benchmark-index", type=int, default=0)
    parser.add_argument("--ssh-ready-timeout-sec", type=float, default=300.0)
    parser.add_argument("--startup-timeout-sec", type=float, default=900.0)
    parser.add_argument("--status-interval-sec", type=float, default=15.0)
    parser.add_argument("--exec-timeout-sec", type=float, default=120.0)
    parser.add_argument("--diagnostic-command", action="append", dest="diagnostic_commands")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper()), format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def configure_imports(nel_next_root: Path) -> None:
    src = nel_next_root / "src"
    if not src.is_dir():
        raise FileNotFoundError(f"NEL src directory not found: {src}")
    sys.path.insert(0, str(src))


def load_env_file(path: Path) -> None:
    if not path.is_file():
        logging.info("No env file found at %s; using current environment", path)
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if sep != "=":
            raise ValueError(f"invalid env line in {path}: {raw_line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def load_eval_config(path: Path) -> Any:
    import yaml
    from nemo_evaluator.config.eval_config import parse_eval_config

    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return parse_eval_config(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


def resolve_env_class(benchmark_name: str) -> Any:
    from nemo_evaluator.benchmarks.terminal_bench_hard import TerminalBenchHard, TerminalBenchHardAASplit

    if benchmark_name == "terminal-bench-hard":
        return TerminalBenchHard
    if benchmark_name == "terminal-bench-hard-aa-split":
        return TerminalBenchHardAASplit
    raise ValueError("this debug script expects terminal-bench-hard or terminal-bench-hard-aa-split")


async def resolve_seed(benchmark_name: str, task_id: str) -> tuple[Any, int, Path]:
    env = resolve_env_class(benchmark_name)()
    matches = [(idx, task_dir) for idx, task_dir in enumerate(env._tasks) if task_dir.name == task_id]
    if len(matches) != 1:
        known = [task_dir.name for task_dir in env._tasks]
        raise RuntimeError(f"expected one task named {task_id!r}, got {len(matches)}; known tasks: {known}")
    task_idx, task_dir = matches[0]
    return await env.seed(task_idx), task_idx, task_dir


def build_ecs_config(
    eval_config: Any, benchmark_index: int, ssh_timeout: float, startup_timeout: float
) -> tuple[Any, Any, Any]:
    from nemo_evaluator.config.sandboxes import EcsFargateSandbox as EcsFargateSandboxConfig
    from nemo_evaluator.orchestration.orchestrator import _build_ecs_sandbox_config

    bench = eval_config.benchmarks[benchmark_index]
    sandbox_cfg = bench.sandbox
    if not isinstance(sandbox_cfg, EcsFargateSandboxConfig):
        raise TypeError(f"benchmark sandbox is not ecs_fargate: {type(sandbox_cfg).__name__}")

    ecs_config = _build_ecs_sandbox_config(sandbox_cfg)
    if ecs_config.ssh_sidecar is None:
        raise RuntimeError("resolved ECS config has no ssh_sidecar")
    return (
        bench,
        sandbox_cfg,
        replace(
            ecs_config,
            startup_timeout_sec=startup_timeout,
            ssh_sidecar=replace(ecs_config.ssh_sidecar, ssh_ready_timeout_sec=ssh_timeout),
        ),
    )


def resolve_sandbox_spec(seed: Any, sandbox_cfg: Any) -> Any:
    from nemo_evaluator.sandbox.base import SandboxSpec

    base = seed.sandbox_spec
    if base is None:
        raise RuntimeError("seed did not provide a sandbox_spec")

    image = base.image
    if image_template := getattr(sandbox_cfg, "image_template", None):
        image = image_template.format_map(seed.metadata)
    if default_image := getattr(sandbox_cfg, "image", None):
        image = default_image

    return SandboxSpec(
        image=image,
        workdir=base.workdir,
        env={**dict(getattr(sandbox_cfg, "container_env", {}) or {}), **dict(base.env)},
        files=dict(base.files),
        entrypoint=base.entrypoint,
        volumes=list(base.volumes),
        environment_dir=base.environment_dir,
    )


def expected_ecr_image(spec: Any, ecs_config: Any) -> str | None:
    from nemo_evaluator.sandbox.ecs_fargate import ImageBuilder, _sanitize_id

    if not (ecs_config.ecr_repository and spec.environment_dir):
        return None
    tag = ImageBuilder.get_ecr_image_tag(spec.environment_dir, _sanitize_id(spec.image or "sandbox"))
    return f"{ecs_config.ecr_repository}:{tag}"


def emit(title: str, payload: Any) -> None:
    print(f"\n== {title} ==", flush=True)
    print(json.dumps(payload, indent=2, sort_keys=True, default=str), flush=True)


def describe_task(sandbox: Any) -> dict[str, Any] | None:
    if sandbox._ecs is None or sandbox._task_arn is None:
        return None
    tasks = sandbox._ecs.describe_tasks(cluster=sandbox._cfg.cluster, tasks=[sandbox._task_arn]).get("tasks") or []
    if not tasks:
        return None
    task = tasks[0]
    return {
        "taskArn": task.get("taskArn"),
        "lastStatus": task.get("lastStatus"),
        "stoppedReason": task.get("stoppedReason"),
        "stopCode": task.get("stopCode"),
        "containers": [
            {
                "name": container.get("name"),
                "lastStatus": container.get("lastStatus"),
                "exitCode": container.get("exitCode"),
                "reason": container.get("reason"),
            }
            for container in task.get("containers", [])
        ],
    }


def make_wait_for_ssh_ready(sandbox: Any, status_interval_sec: float) -> Any:
    def wait_for_ssh_ready(host: str, port: int, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        started = time.monotonic()
        last_status = 0.0
        last_error = None
        print(f"Waiting for SSH at {host}:{port} for up to {timeout:.0f}s", flush=True)

        while time.monotonic() < deadline:
            try:
                with socket.create_connection((host, port), timeout=5.0) as sock:
                    sock.settimeout(5.0)
                    if b"SSH" in sock.recv(256):
                        print(f"SSH ready at {host}:{port} after {time.monotonic() - started:.1f}s", flush=True)
                        return
            except OSError as exc:
                last_error = str(exc)

            now = time.monotonic()
            if now - last_status >= status_interval_sec:
                summary = describe_task(sandbox)
                emit(
                    "ssh wait status",
                    {"elapsed_sec": round(now - started, 1), "last_error": last_error, "task": summary},
                )
                if summary and summary.get("lastStatus") == "STOPPED":
                    raise RuntimeError(f"ECS task stopped while waiting for SSH: {summary}")
                last_status = now
            time.sleep(2.0)

        raise TimeoutError(f"SSH not ready at {host}:{port} after {timeout:.0f}s")

    return wait_for_ssh_ready


async def run_diagnostics(sandbox: Any, commands: list[str], timeout_sec: float) -> None:
    for command in commands:
        result = await sandbox.exec(command, timeout_sec=timeout_sec)
        emit(f"exec: {command}", {"return_code": result.return_code, "stdout": result.stdout, "stderr": result.stderr})
        if result.return_code != 0:
            raise RuntimeError(f"diagnostic failed: {command}")


async def async_main() -> int:
    from nemo_evaluator.sandbox.ecs_fargate import EcsFargateSandbox

    args = parse_args()
    configure_logging(args.log_level)
    configure_imports(args.nel_next_root)
    if args.datasets_dir:
        os.environ["HARBOR_DATASETS_DIR"] = str(args.datasets_dir)
    load_env_file(args.env_file)

    bench, sandbox_cfg, ecs_config = build_ecs_config(
        load_eval_config(args.config),
        args.benchmark_index,
        args.ssh_ready_timeout_sec,
        args.startup_timeout_sec,
    )
    seed, task_idx, task_dir = await resolve_seed(bench.name, args.task_id)
    spec = resolve_sandbox_spec(seed, sandbox_cfg)
    emit(
        "resolved target",
        {
            "benchmark": bench.name,
            "task_id": args.task_id,
            "task_index": task_idx,
            "task_dir": str(task_dir),
            "environment_dir": spec.environment_dir,
            "expected_ecr_image": expected_ecr_image(spec, ecs_config),
        },
    )
    if args.dry_run:
        return 0

    sandbox = EcsFargateSandbox(spec, ecs_config=ecs_config)
    sandbox._wait_for_ssh_ready = make_wait_for_ssh_ready(sandbox, args.status_interval_sec)
    started = False
    try:
        await sandbox.start()
        started = True
        emit(
            "started sandbox",
            {
                "task_arn": sandbox._task_arn,
                "task_definition_arn": sandbox._task_def_arn,
                "task_ip": sandbox._task_ip,
                "local_exec_port": sandbox.local_port,
            },
        )
        await run_diagnostics(sandbox, args.diagnostic_commands or DEFAULT_DIAGNOSTICS, args.exec_timeout_sec)
        return 0
    finally:
        if started:
            await sandbox.stop()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(async_main()))
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
