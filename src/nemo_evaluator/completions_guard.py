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
"""Guard against chat-template controls leaking onto text completions.

``chat_template_kwargs`` / ``chat_template`` only take effect on a chat
endpoint, where the server renders the prompt from ``messages``. The text
completions endpoint (``/v1/completions``) consumes an already-rendered
``prompt`` and never applies a chat template, so these fields are inert: a
toggle such as ``chat_template_kwargs.enable_thinking=false`` injected through
the adapter never reaches the model. That produces a run that *looks* correctly
configured but silently scores with the wrong generation mode.

The policy is tiered and model-agnostic — it never needs to know a model's
specific reasoning key:

* **present (any value)** -> strip the field and log a warning. Removes the
  silent inert field so it can never quietly change (or fail to change) a run.
* **disable intent** (a boolean ``False`` anywhere inside
  ``chat_template_kwargs``, e.g. ``enable_thinking: false`` / ``thinking:
  false``) -> hard error. Someone is trying to turn model behavior *off* in a
  place that is silently ignored, so the behavior would stay *on* and scores
  would be misleading. Failing fast is safer than a misconfigured run.
* ``NEMO_EVALUATOR_STRICT_COMPLETIONS=1`` -> escalate *any* present field
  (even enabling ones) to a hard error, for CI.

A boolean ``False`` is the model-agnostic signal for "turn this off": it does
not matter whether the key is ``enable_thinking``, ``thinking``, ``reasoning``,
etc. String-encoded disables (e.g. ``thinking_mode: "chat"``) are not treated
as a hard error and degrade to strip+warn — still safe, never silent.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHAT_TEMPLATE_ONLY_FIELDS: frozenset[str] = frozenset({"chat_template_kwargs", "chat_template"})

STRICT_ENV_VAR = "NEMO_EVALUATOR_STRICT_COMPLETIONS"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def strict_completions_enabled() -> bool:
    """True when ``NEMO_EVALUATOR_STRICT_COMPLETIONS`` requests hard failures."""
    return os.environ.get(STRICT_ENV_VAR, "").strip().lower() in _TRUTHY


def is_text_completions_path(path: str) -> bool:
    """True for the OpenAI *text* completions endpoint (``/v1/completions``).

    The chat endpoint ``/v1/chat/completions`` also ends in ``/completions``,
    so it is explicitly excluded.
    """
    clean = path.rstrip("/")
    return clean.endswith("/completions") and not clean.endswith("/chat/completions")


def is_text_completions_endpoint_type(endpoint_type: str | None) -> bool:
    """True for the legacy ``endpoint_type: completions`` selector."""
    return (endpoint_type or "").strip().lower() == "completions"


def _has_boolean_false(obj: Any) -> bool:
    """True when a boolean ``False`` appears anywhere inside *obj*.

    ``is False`` (not ``== False``) so the integer ``0`` is not mistaken for a
    disable toggle.
    """
    if obj is False:
        return True
    if isinstance(obj, dict):
        return any(_has_boolean_false(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_boolean_false(item) for item in obj)
    return False


def _scan_chat_template_fields(obj: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    """Find chat-template-only keys inside *obj*.

    Returns ``(locations, disabling_locations)`` as dotted paths.
    ``disabling_locations`` are the ``chat_template_kwargs`` occurrences that
    carry a boolean ``False`` (a disable intent).
    """
    locations: list[str] = []
    disabling: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in CHAT_TEMPLATE_ONLY_FIELDS:
                locations.append(path)
                if key == "chat_template_kwargs" and _has_boolean_false(value):
                    disabling.append(path)
            else:
                sub_locations, sub_disabling = _scan_chat_template_fields(value, path)
                locations.extend(sub_locations)
                disabling.extend(sub_disabling)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            sub_locations, sub_disabling = _scan_chat_template_fields(item, f"{prefix}[{idx}]")
            locations.extend(sub_locations)
            disabling.extend(sub_disabling)
    return locations, disabling


def _strip_chat_template_fields(obj: Any) -> Any:
    """Return a deep copy of *obj* with chat-template-only keys removed."""
    if isinstance(obj, dict):
        return {
            key: _strip_chat_template_fields(value)
            for key, value in obj.items()
            if key not in CHAT_TEMPLATE_ONLY_FIELDS
        }
    if isinstance(obj, list):
        return [_strip_chat_template_fields(item) for item in obj]
    return obj


def _build_message(
    locations: list[str],
    *,
    context: str,
    disabling: list[str],
    strict: bool,
) -> str:
    fields = sorted({loc.rsplit(".", 1)[-1] for loc in locations})
    where = f" (found at: {', '.join(sorted(set(locations)))})"
    base = (
        f"Chat-template-only field(s) [{', '.join(fields)}] present in {context}{where}. "
        "The text completions endpoint (/v1/completions) consumes an already-rendered "
        "'prompt' and never applies a chat template, so these fields are inert and never "
        "reach the model."
    )
    guidance = (
        " Configure the prompt-side toggle where the prompt is rendered client-side instead "
        "(for nemo-skills harnesses pass it via params.extra.args, e.g. "
        "'++chat_template_kwargs.enable_thinking=false'; the exact key is model-specific), "
        "or use a /v1/chat/completions endpoint for server-side templating."
    )
    if disabling:
        why = (
            f" A disabling value (boolean false) is set under {', '.join(sorted(set(disabling)))} — "
            "an attempt to turn model behavior (e.g. reasoning) OFF that is silently ignored here, "
            "so the model would run with it still ON and scores would be misleading. Failing fast."
        )
        return base + why + guidance
    if strict:
        why = f" {STRICT_ENV_VAR} is set, so any chat-template field on a completions endpoint is a hard error."
        return base + why + guidance
    why = (
        " They were stripped before reaching the model. A disabling value (boolean false) here would "
        f"be a hard error; set {STRICT_ENV_VAR}=1 to make any such field a hard error."
    )
    return base + why + guidance


def enforce_text_completions_body(path: str, body: dict[str, Any], *, strict: bool | None = None) -> None:
    """Enforce the chat-template policy on a text-completions request *body*.

    Mutates *body* in place (strips inert fields). No-op for chat endpoints or
    clean bodies. Raises ``ValueError`` when a disable intent is present, or for
    any present field when strict mode is enabled.
    """
    if not is_text_completions_path(path):
        return
    locations, disabling = _scan_chat_template_fields(body)
    if not locations:
        return
    strict = strict_completions_enabled() if strict is None else strict
    message = _build_message(locations, context=f"a request to {path!r}", disabling=disabling, strict=strict)
    if disabling or strict:
        raise ValueError(message)
    logger.warning(message)
    for name in [key for key in body if key in CHAT_TEMPLATE_ONLY_FIELDS]:
        body.pop(name, None)


def normalize_adapter_config_for_endpoint(
    adapter_config: dict[str, Any] | None,
    endpoint_type: str | None,
    *,
    strict: bool | None = None,
) -> dict[str, Any] | None:
    """Enforce the chat-template policy on a legacy ``adapter_config``.

    Only acts when *endpoint_type* selects the text completions endpoint and
    the config carries a chat-template-only field (e.g.
    ``params_to_add.chat_template_kwargs``). Returns the config with inert
    fields stripped, or the input unchanged when there is nothing to do. Raises
    ``ValueError`` when a disable intent is present, or for any present field
    when strict mode is enabled.
    """
    if not adapter_config:
        return adapter_config
    if not is_text_completions_endpoint_type(endpoint_type):
        return adapter_config
    locations, disabling = _scan_chat_template_fields(adapter_config)
    if not locations:
        return adapter_config
    strict = strict_completions_enabled() if strict is None else strict
    message = _build_message(
        locations,
        context="adapter_config for a completions endpoint",
        disabling=disabling,
        strict=strict,
    )
    if disabling or strict:
        raise ValueError(message)
    logger.warning(message)
    return _strip_chat_template_fields(adapter_config)
