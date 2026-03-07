"""Adapter for legacy NeMo Evaluator containers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_TAG = os.environ.get("NEL_IMAGE_TAG", "26.01")

HARNESS_REPOS: dict[str, str] = {
    "simple_evals": "nvcr.io/nvidia/eval-factory/simple-evals",
    "lm-evaluation-harness": "nvcr.io/nvidia/eval-factory/lm-evaluation-harness",
    "nemo_skills": "nvcr.io/nvidia/eval-factory/nemo-skills",
    "mtbench": "nvcr.io/nvidia/eval-factory/mtbench",
    "bfcl": "nvcr.io/nvidia/eval-factory/bfcl",
    "hle": "nvcr.io/nvidia/eval-factory/hle",
    "ifbench": "nvcr.io/nvidia/eval-factory/ifbench",
    "livecodebench": "nvcr.io/nvidia/eval-factory/livecodebench",
    "scicode": "nvcr.io/nvidia/eval-factory/scicode",
    "bigcode-evaluation-harness": "nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness",
    "vlmevalkit": "nvcr.io/nvidia/eval-factory/vlmevalkit",
    "garak": "nvcr.io/nvidia/eval-factory/garak",
    "safety_eval": "nvcr.io/nvidia/eval-factory/safety-harness",
    "helm": "nvcr.io/nvidia/eval-factory/helm",
    "tooltalk": "nvcr.io/nvidia/eval-factory/tooltalk",
    "profbench": "nvcr.io/nvidia/eval-factory/profbench",
    "mmath": "nvcr.io/nvidia/eval-factory/mmath",
    "tau2_bench": "nvcr.io/nvidia/eval-factory/tau2-bench",
    "long_context_eval": "nvcr.io/nvidia/eval-factory/long-context-eval",
    "mteb": "nvcr.io/nvidia/eval-factory/mteb",
    "aa-lcr": "nvcr.io/nvidia/eval-factory/aa-lcr",
}


def get_harness_image(harness: str, tag: str | None = None) -> str:
    tag = tag or DEFAULT_IMAGE_TAG
    if harness not in HARNESS_REPOS:
        raise ValueError(f"Unknown harness {harness!r}. Known: {sorted(HARNESS_REPOS)}")
    return f"{HARNESS_REPOS[harness]}:{tag}"


HARNESS_IMAGES = {k: f"{v}:{DEFAULT_IMAGE_TAG}" for k, v in HARNESS_REPOS.items()}


@dataclass
class ContainerConfig:
    task_type: str
    model_url: str
    model_id: str
    api_key_env: str = "NEMO_API_KEY"
    harness: str | None = None
    image: str | None = None
    image_tag: str | None = None
    extra_params: dict[str, Any] = field(default_factory=dict)
    limit_samples: int | None = None
    temperature: float = 0.0
    max_new_tokens: int | None = None
    docker_args: list[str] = field(default_factory=list)
    timeout: int = 3600


def _resolve_image(cfg: ContainerConfig) -> str:
    if cfg.image:
        return cfg.image
    harness = cfg.harness or cfg.task_type.split(".")[0]
    return get_harness_image(harness, tag=cfg.image_tag)


def _build_run_config(cfg: ContainerConfig) -> dict[str, Any]:
    task = cfg.task_type.split(".")[-1] if "." in cfg.task_type else cfg.task_type
    run_cfg: dict[str, Any] = {
        "config": {
            "type": task,
            "output_dir": "/results",
            "params": {
                "temperature": cfg.temperature,
                **({} if cfg.limit_samples is None else {"limit_samples": cfg.limit_samples}),
                **({} if cfg.max_new_tokens is None else {"max_new_tokens": cfg.max_new_tokens}),
                "extra": cfg.extra_params,
            },
        },
        "target": {
            "api_endpoint": {
                "url": cfg.model_url,
                "model_id": cfg.model_id,
                "api_key_name": cfg.api_key_env,
            },
        },
    }
    return run_cfg


def _parse_results_yml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Container did not produce results.yml at {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _parse_metrics_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _extract_scores(results_data: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    results = results_data.get("results", {})

    for task_name, task_data in results.get("tasks", {}).items():
        metrics = task_data.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            metric_scores = metric_data.get("scores", {})
            for agg_name, agg_data in metric_scores.items():
                if isinstance(agg_data, dict) and "value" in agg_data:
                    key = f"{task_name}/{metric_name}/{agg_name}"
                    scores[key] = {
                        "value": agg_data["value"],
                        "stats": agg_data.get("stats", {}),
                    }

    for group_name, group_data in results.get("groups", {}).items():
        metrics = group_data.get("metrics", {})
        for metric_name, metric_data in metrics.items():
            metric_scores = metric_data.get("scores", {})
            for agg_name, agg_data in metric_scores.items():
                if isinstance(agg_data, dict) and "value" in agg_data:
                    key = f"group:{group_name}/{metric_name}/{agg_name}"
                    scores[key] = {
                        "value": agg_data["value"],
                        "stats": agg_data.get("stats", {}),
                    }

    return scores


def run_container_eval(cfg: ContainerConfig, output_dir: str | None = None) -> dict[str, Any]:
    image = _resolve_image(cfg)
    run_config = _build_run_config(cfg)

    host_dir = Path(output_dir or tempfile.mkdtemp(prefix="nel_container_"))
    host_dir.mkdir(parents=True, exist_ok=True)

    config_path = host_dir / "config_ef.yaml"
    with open(config_path, "w") as f:
        yaml.dump(run_config, f)

    api_key = os.environ.get(cfg.api_key_env, "")
    cmd = [
        "docker", "run", "--rm",
        "-v", f"{host_dir}:/results",
        "-v", f"{config_path}:/config_ef.yaml:ro",
        "-e", f"{cfg.api_key_env}={api_key}",
        *cfg.docker_args,
        image,
        "nemo-evaluator", "run_eval", "--run_config", "/config_ef.yaml",
    ]

    logger.info("Running container: %s (image=%s task=%s)", cfg.task_type, image, cfg.task_type)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=cfg.timeout)

    if result.returncode != 0:
        logger.error("Container failed (exit=%d): %s", result.returncode, result.stderr[-2000:])
        raise RuntimeError(f"Container eval failed with exit code {result.returncode}")

    results_data = _parse_results_yml(host_dir / "results.yml")
    metrics_data = _parse_metrics_json(host_dir / "eval_factory_metrics.json")
    scores = _extract_scores(results_data)

    response_stats = metrics_data.get("response_stats", {})
    eval_timing = metrics_data.get("evaluation", {})
    reasoning_stats = metrics_data.get("reasoning", {})

    bundle: dict[str, Any] = {
        "source": "container",
        "image": image,
        "task": cfg.task_type,
        "model": cfg.model_id,
        "config": run_config,
        "scores": scores,
        "runtime": {
            "elapsed_seconds": eval_timing.get("runtime_seconds", 0),
            "inference_time_seconds": eval_timing.get("inference_time_seconds", 0),
            "scoring_time_seconds": eval_timing.get("scoring_time_seconds", 0),
        },
        "response_stats": {
            "avg_latency_ms": response_stats.get("avg_latency_ms", 0),
            "avg_prompt_tokens": response_stats.get("avg_prompt_tokens", 0),
            "avg_completion_tokens": response_stats.get("avg_completion_tokens", 0),
            "count": response_stats.get("count", 0),
            "successful_count": response_stats.get("successful_count", 0),
            "finish_reasons": response_stats.get("finish_reason", {}),
        },
        "reasoning": reasoning_stats,
        "results_yml_path": str(host_dir / "results.yml"),
        "metrics_json_path": str(host_dir / "eval_factory_metrics.json"),
    }

    logger.info("Container eval complete: %d scores extracted", len(scores))
    return bundle


def list_harnesses() -> list[dict[str, str]]:
    return [{"harness": h, "image": i} for h, i in sorted(HARNESS_IMAGES.items())]
