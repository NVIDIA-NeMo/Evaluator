"""Resolve container images for SLURM execution.

Reads the bundled containers.toml mapping and resolves which eval container
a given benchmark name requires. Users can override images via
``cluster.container_overrides`` in their config.
"""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    from nemo_evaluator.resources import containers_toml_path
    return tomllib.loads(containers_toml_path().read_text(encoding="utf-8"))


def _expand(template: str, defaults: dict[str, str]) -> str:
    """Expand {registry} and {tag} placeholders."""
    return template.format_map(defaults)


def resolve_eval_image(
    bench_name: str,
    *,
    overrides: dict[str, str] | None = None,
    tag: str | None = None,
) -> str:
    """Return the container image for a benchmark.

    Args:
        bench_name: Benchmark URI, e.g. ``lm-eval://aime2025``.
        overrides: Per-scheme overrides from ``cluster.container_overrides``.
        tag: Override the default image tag.
    """
    mapping = _load_mapping()
    defaults = dict(mapping.get("defaults", {}))
    if tag:
        defaults["tag"] = tag

    schemes = mapping.get("schemes", {})
    eval_images = mapping.get("eval", {})

    variant = "base"
    for scheme, var in schemes.items():
        if bench_name.startswith(scheme):
            variant = var
            break

    if overrides and variant in overrides:
        return overrides[variant]

    if not variant:
        return ""

    template = eval_images.get(variant, eval_images.get("base", ""))
    return _expand(template, defaults)


def resolve_deployment_image(deploy_type: str) -> str:
    """Return the default deployment container for a server type."""
    mapping = _load_mapping()
    return mapping.get("deployment", {}).get(deploy_type, "")


def list_variants() -> dict[str, str]:
    """Return all eval container variants with their expanded image names."""
    mapping = _load_mapping()
    defaults = dict(mapping.get("defaults", {}))
    eval_images = mapping.get("eval", {})
    return {k: _expand(v, defaults) for k, v in eval_images.items()}
