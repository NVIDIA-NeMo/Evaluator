# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write-once snapshot of the composed run config for reproducibility.

Every generated artifact in a run directory (sbatch scripts, per-shard
``config_*.yaml`` execution configs, secrets env) is a deterministic
function of two inputs: the composed config and the nemo-evaluator
version. The per-benchmark execution configs deliberately transform the
config (managed deployments become ``${ENV_VAR}`` API stubs, cluster is
forced to ``local``), so they cannot serve as a reproducibility record.
This module persists the root of that derivation tree instead: the
config as composed from fragments with CLI overrides applied, but with
``${ENV_VAR}`` references intentionally NOT expanded — so secrets never
reach disk by construction. A provenance header records the package
version, which pins all schema defaults.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

SNAPSHOT_FILENAME = "full_config.yaml"

_MASK = "<redacted>"

# Field names whose literal (non-``${...}``) values are always masked.
# Plural forms are matched too, except 'tokens': benign *_tokens params
# (max_new_tokens, tokens_to_generate) must not be masked.
_SECRET_KEY_RE = re.compile(
    r"(api[_-]?keys?|token|secrets?|passwords?|passwd|credentials?|authorization)(?![A-Za-z])", re.IGNORECASE
)

# Backstop for secret-looking literals hardcoded anywhere (including
# inside free-form strings such as ``extra.args``).
_SECRET_VALUE_RES = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-]{16,}"),
]


def package_version() -> str:
    """Best-effort nemo-evaluator package version."""
    try:
        from importlib.metadata import version

        return version("nemo-evaluator")
    except Exception:  # pragma: no cover - metadata missing in odd installs
        return "unknown"


# ``key=value`` tokens (CLI overrides, argv) whose key looks secret.
_SECRET_KV_RE = re.compile(
    r"(?P<key>[A-Za-z0-9_.\-]*(?:api[_-]?keys?|token|secrets?|passwords?|passwd|credentials?|authorization)(?![A-Za-z])[A-Za-z0-9_.\-]*\s*=\s*)(?P<value>[^\s'\"]+)",
    re.IGNORECASE,
)

# Space-separated secret flags, e.g. quick mode's ``--api-key <value>``.
_SECRET_FLAG_RE = re.compile(
    r"(?P<key>--?[A-Za-z0-9\-]*(?:api[_-]?keys?|token|secrets?|passwords?|passwd|credentials?|authorization)(?![A-Za-z])[A-Za-z0-9\-]*\s+)(?P<value>[^\s'\"-][^\s'\"]*)",
    re.IGNORECASE,
)


# Values that are references, not secrets: ${VAR} (anywhere), bare $VAR,
# an env-var NAME (all-caps identifier — the EF convention for "read this
# env var", e.g. ``api_key: JUDGE_API_KEY``), or a v1 ``host:VAR`` mapping.
# Trade-off: an actual secret shaped exactly like an all-caps identifier
# would be kept, but real keys (nvapi-.., mixed case, dashes) never are —
# and the value-pattern backstop still applies.
_REFERENCE_VALUE_RES = [
    re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$"),
    re.compile(r"^(host:)?[A-Z][A-Z0-9_]*$"),
]


def _is_reference_value(value: str) -> bool:
    return "${" in value or any(p.fullmatch(value) for p in _REFERENCE_VALUE_RES)


def _maskable_secret_literal(value: str) -> bool:
    """Whether a value under a secret-named key is plausibly a secret literal.

    Empty strings and values with whitespace or ``<>|`` (template markers
    like adapter ``start_reasoning_token: '<|channel>thought'``) are not
    secrets; references (env names, ``$VAR``/``${VAR}``) are exempt too.
    """
    if not value or _is_reference_value(value):
        return False
    # Paths and AWS ARNs are pointers, not secret literals (tiktoken paths
    # otherwise trip the 'token' key match; secret ARNs *reference* secrets).
    if value.startswith(("/", "./", "~/", "arn:")):
        return False
    return not re.search(r"[\s<>|]", value)


def _mask_string(value: str) -> str:
    for pattern in _SECRET_VALUE_RES:
        value = pattern.sub(_MASK, value)
    return value


def _mask_cli_text(text: str) -> str:
    """Mask secret values in CLI-shaped text (``--api-key x``, ``-O a.api_key=x``).

    ``${VAR}`` values are kept — they are the safe representation.
    """

    def _kv_sub(m: re.Match) -> str:
        if _is_reference_value(m.group("value")):
            return m.group(0)
        return m.group("key") + _MASK

    text = _SECRET_KV_RE.sub(_kv_sub, text)
    text = _SECRET_FLAG_RE.sub(_kv_sub, text)
    return _mask_string(text)


def mask_secrets(obj: Any, *, key: str | None = None) -> Any:
    """Return a copy of *obj* with secret literals masked.

    ``${ENV_VAR}`` references are preserved untouched — they are the
    safe, intended representation. Only literal values are masked:
    any string under a secret-looking key, plus any string anywhere
    matching a known key/token pattern.
    """
    if isinstance(obj, dict):
        return {k: mask_secrets(v, key=str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [mask_secrets(v, key=key) for v in obj]
    if isinstance(obj, str):
        if key is not None and _SECRET_KEY_RE.search(key) and _maskable_secret_literal(obj):
            return _MASK
        return _mask_string(obj)
    return obj


def build_provenance(
    source_config: str | None = None,
    overrides: tuple[str, ...] | list[str] = (),
    run_id: str | None = None,
) -> dict[str, str]:
    """Provenance metadata recorded in the snapshot header."""
    prov: dict[str, str] = {
        "nemo-evaluator version": package_version(),
        "created": datetime.now(timezone.utc).isoformat(),
        "command": _mask_cli_text(shlex.join(sys.argv)),
    }
    if source_config:
        prov["source config"] = source_config
    if overrides:
        prov["overrides"] = _mask_cli_text(" ".join(overrides))
    if run_id:
        prov["run_id"] = run_id
    return prov


def build_snapshot_text(raw: dict[str, Any], provenance: dict[str, str]) -> str:
    """Render the snapshot file: provenance header + masked config YAML."""
    lines = [
        "# Re-run with:  nel eval run <this file>",
    ]
    for k, v in provenance.items():
        lines.append(f"# {k}: {v}")
    header = "\n".join(lines) + "\n"
    body = yaml.dump(mask_secrets(raw), default_flow_style=False, sort_keys=False)
    return header + body


def record_output_dir_override(config: Any, output_dir: str) -> None:
    """Keep the composed snapshot consistent with a CLI ``-o`` override.

    The override mutates ``config.output.dir`` after the raw composed dict
    was captured; without this the snapshot would record the YAML's
    original output dir and re-running it would write elsewhere.
    """
    raw = getattr(config, "_composed_raw", None)
    if isinstance(raw, dict):
        raw.setdefault("output", {})["dir"] = output_dir


def write_config_snapshot(config: Any, output_dir: str | Path | None = None, *, force: bool = False) -> Path | None:
    """Write the composed-config snapshot into *output_dir*.

    An existing snapshot is preserved unless *force* is set: resumes must
    keep the original submission record, while a fresh run into a reused
    output directory (common for local runs without timestamped subdirs)
    must overwrite it — a stale "full config" is worse than none.

    Skipped inside shard/inner executions (``NEL_INNER_EXECUTION=1``) —
    those load the machine-generated per-benchmark config, which must
    not be recorded as the full run config. Never raises: snapshotting
    is an observability feature and must not break a run.
    """
    if os.environ.get("NEL_INNER_EXECUTION") == "1":
        return None
    try:
        out_dir = Path(output_dir or config.output.dir)
        path = out_dir / SNAPSHOT_FILENAME
        if path.exists() and not force:
            logger.debug("Config snapshot already exists (kept, force=False): %s", path)
            return path

        raw = getattr(config, "_composed_raw", None)
        provenance = dict(getattr(config, "_snapshot_provenance", None) or {})
        if raw is None:
            # Quick mode / programmatic configs: reconstruct from the
            # validated model. Defaults are inlined; secrets are masked.
            raw = config.model_dump(mode="json", exclude_none=True)
            provenance.setdefault("note", "reconstructed from validated config (no source YAML)")
        if not provenance:
            provenance = build_provenance()

        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(build_snapshot_text(raw, provenance), encoding="utf-8")
        logger.info("Saved run config snapshot: %s", path)
        return path
    except Exception as exc:  # noqa: BLE001 - never break a run over the snapshot
        logger.warning("Could not write run config snapshot: %s", exc)
        return None
