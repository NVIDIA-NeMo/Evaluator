# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo Evaluator runner for the RASB 26H1 benchmark snapshot."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger("nemo_evaluator.rasb_26h1")

DEFAULT_RASB_ROOT = Path("/rasb-26h1")
DEFAULT_ENVIRONMENTS_DIR = "26h1"
DEFAULT_BENCHMARK_WORKERS = 2
DEFAULT_BENCHMARK_TIMEOUT = 3600


def _parse_slice(spec: str) -> tuple[int | None, int | None]:
    if ":" in spec:
        start_s, end_s = spec.split(":", 1)
        start = int(start_s) if start_s else None
        end = int(end_s) if end_s else None
        return start, end
    return 0, int(spec)


def _strip_endpoint_path(url: str | None) -> str | None:
    if not url:
        return None
    stripped = url.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if stripped.endswith(suffix):
            return stripped[: -len(suffix)]
    return stripped


def _get_api_key(api_key_name: str | None) -> str | None:
    if not api_key_name:
        return None
    return os.environ.get(api_key_name)


def _resolve_rasb_root(rasb_root: str | None) -> Path:
    if rasb_root:
        root = Path(rasb_root)
    elif os.environ.get("NEMO_EVALUATOR_DATASET_DIR"):
        root = Path(os.environ["NEMO_EVALUATOR_DATASET_DIR"])
    elif os.environ.get("NEMO_EVALUATOR_RASB_ROOT"):
        root = Path(os.environ["NEMO_EVALUATOR_RASB_ROOT"])
    else:
        root = DEFAULT_RASB_ROOT

    root = root.expanduser().resolve()
    if not (root / "benchmark.py").exists():
        raise FileNotFoundError(
            f"RASB 26H1 root is not valid: {root}. "
            "Expected benchmark.py under this directory. Set "
            "config.params.extra.rasb_root or mount the snapshot as "
            "NEMO_EVALUATOR_DATASET_DIR."
        )
    return root


def _resolve_path(root: Path, value: str | None, default: str | None = None) -> Path:
    raw = value if value is not None else default
    if raw is None:
        raise ValueError("No path value provided")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _load_rasb_benchmark(root: Path) -> ModuleType:
    benchmark_path = root / "benchmark.py"
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    spec = importlib.util.spec_from_file_location(
        "nemo_evaluator_rasb_26h1_benchmark", benchmark_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RASB benchmark module: {benchmark_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _select_callable(root: Path, explicit_callable: str | None, model_id: str) -> Path:
    if explicit_callable:
        return _resolve_path(root, explicit_callable)

    model_l = model_id.lower()
    callable_name = (
        "anthropic_lm"
        if "anthropic" in model_l or "claude" in model_l
        else "openai_lm"
    )
    return root / "callables" / callable_name


def _discover_env_dirs(
    environments_dir: Path,
    *,
    only: str | None = None,
    slice_spec: str | None = None,
    limit_samples: int | None = None,
) -> list[tuple[str, Path]]:
    env_dirs = [
        (path.name, path)
        for path in sorted(environments_dir.iterdir())
        if path.is_dir() and (path / "metadata.json").exists()
    ]
    if only:
        env_dirs = [(env_id, path) for env_id, path in env_dirs if only in env_id]

    effective_slice = slice_spec
    if effective_slice is None and limit_samples is not None:
        effective_slice = str(limit_samples)
    if effective_slice is not None:
        start, end = _parse_slice(effective_slice)
        env_dirs = env_dirs[start:end]
    return env_dirs


def _build_extra_env(args: argparse.Namespace, api_key: str | None) -> dict[str, str]:
    extra_env: dict[str, str] = {"TARGET_MODEL": args.model_id}
    if api_key:
        extra_env.update(
            {
                "NVAPI_KEY": api_key,
                "OPENAI_API_KEY": api_key,
                "ANTHROPIC_API_KEY": api_key,
            }
        )

    base_url = _strip_endpoint_path(args.model_url)
    if base_url:
        extra_env["NVIDIA_BASE_URL"] = base_url
        extra_env["OPENAI_BASE_URL"] = base_url

    if args.temperature is not None:
        extra_env["RASB_TEMPERATURE"] = str(args.temperature)
    if args.judge_model:
        extra_env["JUDGE_MODEL"] = args.judge_model
    return extra_env


def _env_file_for_run(root: Path, requested_env_file: str | None) -> Path:
    if requested_env_file:
        env_file = _resolve_path(root, requested_env_file)
        if not env_file.exists():
            raise FileNotFoundError(f"RASB env file not found: {env_file}")
        return env_file

    tmp_dir = Path(tempfile.mkdtemp(prefix="rasb-26h1-env-"))
    env_file = tmp_dir / ".env"
    env_file.write_text("", encoding="utf-8")
    return env_file


def _load_cached_result(results_file: Path) -> dict[str, Any] | None:
    try:
        return json.loads(results_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _benchmark_one(
    benchmark_module: ModuleType,
    *,
    env_id: str,
    env_dir: Path,
    callable_dir: Path,
    run_name: str,
    results_base: Path,
    env_file: Path,
    timeout: int,
    keep_containers: bool,
    redo_on_error: bool,
    extra_env: dict[str, str],
) -> dict[str, Any]:
    results_dir = results_base / "environments" / env_id
    results_file = results_dir / "results.json"
    inputs_dir = env_dir / "inputs"
    expected_samples = len(list(inputs_dir.glob("synth_*.json")))

    if results_file.exists():
        cached = _load_cached_result(results_file)
        if cached:
            cached_total = int(cached.get("total", 0) or 0)
            cached_errors = int(cached.get("errors", 0) or 0)
            if cached_total >= expected_samples and cached_errors == 0:
                logger.info(
                    "Skipping %s; cached result is complete (%s/%s)",
                    env_id,
                    cached.get("passed", 0),
                    cached_total,
                )
                return cached
            if cached_errors and redo_on_error:
                shutil.rmtree(results_dir, ignore_errors=True)
        else:
            shutil.rmtree(results_dir, ignore_errors=True)

    return benchmark_module.run_in_docker(
        env_id=env_id,
        env_dir=env_dir,
        callable_dir=callable_dir,
        env_file=env_file,
        timeout=timeout,
        run_name=run_name,
        keep_container=keep_containers,
        results_out=results_dir,
        extra_env=extra_env,
    )


def _run_benchmark(
    benchmark_module: ModuleType,
    *,
    env_dirs: list[tuple[str, Path]],
    callable_dir: Path,
    run_name: str,
    results_base: Path,
    env_file: Path,
    workers: int,
    timeout: int,
    keep_containers: bool,
    redo_on_error: bool,
    extra_env: dict[str, str],
) -> list[dict[str, Any]]:
    results_base.mkdir(parents=True, exist_ok=True)
    (results_base / "environments").mkdir(exist_ok=True)

    logger.info(
        "Benchmarking %d RASB environments with %d worker(s)",
        len(env_dirs),
        workers,
    )
    all_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _benchmark_one,
                benchmark_module,
                env_id=env_id,
                env_dir=env_dir,
                callable_dir=callable_dir,
                run_name=run_name,
                results_base=results_base,
                env_file=env_file,
                timeout=timeout,
                keep_containers=keep_containers,
                redo_on_error=redo_on_error,
                extra_env=extra_env,
            ): env_id
            for env_id, env_dir in env_dirs
        }
        for future in as_completed(futures):
            env_id = futures[future]
            try:
                all_results.append(future.result())
            except Exception as exc:
                logger.exception("RASB environment failed: %s", env_id)
                all_results.append(
                    {"env_id": env_id, "status": "error", "error": str(exc)}
                )
    return all_results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run RASB 26H1 from NeMo Evaluator")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task", default="rasb_26h1")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--api-key-name", default=None)
    parser.add_argument("--rasb-root", default=None)
    parser.add_argument("--callable", default=None)
    parser.add_argument("--environments", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--only", default=None)
    parser.add_argument("--slice", default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument(
        "--benchmark-workers",
        type=int,
        default=DEFAULT_BENCHMARK_WORKERS,
    )
    parser.add_argument(
        "--benchmark-timeout",
        type=int,
        default=DEFAULT_BENCHMARK_TIMEOUT,
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--keep-containers", action="store_true", default=False)
    parser.add_argument("--clean", action="store_true", default=False)
    parser.add_argument("--redo-on-error", action="store_true", default=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_arg_parser().parse_args(argv)

    root = _resolve_rasb_root(args.rasb_root)
    benchmark_module = _load_rasb_benchmark(root)

    environments_dir = _resolve_path(root, args.environments, DEFAULT_ENVIRONMENTS_DIR)
    callable_dir = _select_callable(root, args.callable, args.model_id).resolve()
    if not (callable_dir / "callable.py").exists():
        raise FileNotFoundError(f"RASB callable not found: {callable_dir}")

    env_file = _env_file_for_run(root, args.env_file)
    output_dir = Path(args.output_dir).resolve()
    results_base = output_dir / "rasb_results"
    run_name = args.run_name or output_dir.name or "rasb_26h1"

    if args.clean and results_base.exists():
        shutil.rmtree(results_base)

    env_dirs = _discover_env_dirs(
        environments_dir,
        only=args.only,
        slice_spec=args.slice,
        limit_samples=args.limit_samples,
    )
    if not env_dirs:
        raise RuntimeError(f"No RASB environments found in {environments_dir}")

    api_key = _get_api_key(args.api_key_name)
    extra_env = _build_extra_env(args, api_key)
    results = _run_benchmark(
        benchmark_module,
        env_dirs=env_dirs,
        callable_dir=callable_dir,
        run_name=run_name,
        results_base=results_base,
        env_file=env_file,
        workers=max(1, args.benchmark_workers),
        timeout=args.benchmark_timeout,
        keep_containers=args.keep_containers,
        redo_on_error=args.redo_on_error,
        extra_env=extra_env,
    )

    summary = benchmark_module.compute_aggregate(results, run_name, str(callable_dir))
    summary["model"] = args.model_id
    summary["timestamp"] = datetime.now(timezone.utc).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "RASB 26H1 complete: pass_rate=%.4f samples=%d",
        summary["aggregate"]["overall_pass_rate"],
        summary["aggregate"]["total_samples"],
    )


if __name__ == "__main__":
    main()
