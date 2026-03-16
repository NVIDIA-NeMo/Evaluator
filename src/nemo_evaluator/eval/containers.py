"""Resolve container images for SLURM and Docker execution.

Images are resolved from the bundled ``containers.toml`` which maps benchmark
URI schemes to container variants (base, lm-eval, skills, mteb, full).

Override: set ``cluster.container_image`` in config to a base image like
``my-registry.io/nel:v2.0``. Variants are derived automatically by appending
``-lm-eval``, ``-skills``, etc. to the tag.
"""
from __future__ import annotations

import tomllib
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def _load_mapping() -> dict[str, Any]:
    from nemo_evaluator.resources import containers_toml_path
    return tomllib.loads(containers_toml_path().read_text(encoding="utf-8"))


def _default_base() -> str:
    """Return the base image from containers.toml."""
    mapping = _load_mapping()
    defaults = mapping.get("defaults", {})
    return f"{defaults['registry']}:{defaults['tag']}"


def _variant_image(base: str, variant: str) -> str:
    """Derive a variant image from a base image.

    ``my-registry.io/nel:v2`` + ``lm-eval`` → ``my-registry.io/nel:v2-lm-eval``

    Skips appending if the base already ends with the variant suffix.
    """
    if variant == "base" or not variant:
        return base
    if base.endswith(f"-{variant}"):
        return base
    if ":" in base:
        return f"{base}-{variant}"
    return f"{base}:{variant}"


def _scheme_to_variant(bench_name: str) -> str:
    """Map a benchmark URI to its container variant."""
    mapping = _load_mapping()
    schemes = mapping.get("schemes", {})
    for scheme, var in schemes.items():
        if bench_name.startswith(scheme):
            return var
    return "base"


def resolve_eval_image(bench_name: str, base_override: str | None = None) -> str:
    """Return the container image for a benchmark.

    Args:
        bench_name: Benchmark URI, e.g. ``lm-eval://aime2025``.
        base_override: ``cluster.container_image`` — if set, used as-is
            (the user knows which image they want).
    """
    if base_override:
        return base_override
    variant = _scheme_to_variant(bench_name)
    if not variant:
        return ""
    return _variant_image(_default_base(), variant)


def resolve_deployment_image(deploy_type: str) -> str:
    """Return the default deployment container for a server type."""
    mapping = _load_mapping()
    return mapping.get("deployment", {}).get(deploy_type, "")


def default_base_image(base_override: str | None = None) -> str:
    """Return the base eval image."""
    return base_override or _default_base()


