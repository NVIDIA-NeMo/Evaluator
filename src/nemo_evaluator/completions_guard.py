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

The policy is model-agnostic — it never needs to know a model's specific
reasoning key:

* a **boolean toggle** (``True`` or ``False``) anywhere inside
  ``chat_template_kwargs`` (e.g. ``enable_thinking: true`` / ``thinking:
  false``) -> hard error. A boolean is a deliberate attempt to switch model
  behavior on or off; on a text completions endpoint that switch is silently
  ignored, so the run would score in the wrong mode. Failing fast is safer
  than a misconfigured run.
* any other present field (a ``chat_template`` string, non-boolean
  ``chat_template_kwargs`` values such as ``thinking_mode: "chat"``, etc.) ->
  strip the field and log a warning. It is inert here but carries no explicit
  on/off intent, so the run continues without it.
* ``NEMO_EVALUATOR_ALLOW_COMPLETIONS_CHAT_TEMPLATE=1`` -> opt out of the hard
  error. Every chat-template field, booleans included, is stripped with a
  warning instead of failing. Use this when a boolean toggle is inherited from
  a shared config and is known to be safe to ignore on this endpoint.

Why *any* boolean and not only ``False``: model key names are not standardized
(``enable_thinking``, ``disable_thinking``, ``thinking``, ...), so the value
alone cannot distinguish an "on" from an "off" (``disable_thinking: true`` is a
disable expressed as ``True``). Treating any boolean as a deliberate toggle is
the model-agnostic signal that someone tried to control generation in a place
where it is silently ignored.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

CHAT_TEMPLATE_ONLY_FIELDS: frozenset[str] = frozenset({"chat_template_kwargs", "chat_template"})

ALLOW_ENV_VAR = "NEMO_EVALUATOR_ALLOW_COMPLETIONS_CHAT_TEMPLATE"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def allow_completions_chat_template() -> bool:
    """True when ``NEMO_EVALUATOR_ALLOW_COMPLETIONS_CHAT_TEMPLATE`` downgrades the hard error to a warning."""
    return os.environ.get(ALLOW_ENV_VAR, "").strip().lower() in _TRUTHY


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


def _has_boolean(obj: Any) -> bool:
    """True when a boolean (``True`` or ``False``) appears anywhere inside *obj*.

    ``isinstance(..., bool)`` so the integers ``0``/``1`` are not mistaken for a
    toggle (``isinstance(0, bool)`` is ``False``).
    """
    if isinstance(obj, bool):
        return True
    if isinstance(obj, dict):
        return any(_has_boolean(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_has_boolean(item) for item in obj)
    return False


def _scan_chat_template_fields(obj: Any, prefix: str = "") -> tuple[list[str], list[str]]:
    """Find chat-template-only keys inside *obj*.

    Returns ``(locations, boolean_locations)`` as dotted paths.
    ``boolean_locations`` are the ``chat_template_kwargs`` occurrences whose
    value carries a boolean (``True``/``False``) — a deliberate on/off toggle.
    """
    locations: list[str] = []
    boolean_locations: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in CHAT_TEMPLATE_ONLY_FIELDS:
                locations.append(path)
                if key == "chat_template_kwargs" and _has_boolean(value):
                    boolean_locations.append(path)
            else:
                sub_locations, sub_boolean = _scan_chat_template_fields(value, path)
                locations.extend(sub_locations)
                boolean_locations.extend(sub_boolean)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            sub_locations, sub_boolean = _scan_chat_template_fields(item, f"{prefix}[{idx}]")
            locations.extend(sub_locations)
            boolean_locations.extend(sub_boolean)
    return locations, boolean_locations


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
    boolean_locations: list[str],
    fatal: bool,
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
    if fatal:
        why = (
            f" A boolean toggle is set under {', '.join(sorted(set(boolean_locations)))} — "
            "an attempt to switch model behavior (e.g. reasoning) on or off that is silently "
            "ignored here, so the model would run in the wrong mode and scores would be "
            f"misleading. Failing fast. If this was intentional, set {ALLOW_ENV_VAR}=1 to "
            "deactivate this failure (the field is stripped and only a warning is logged)."
        )
        return base + why + guidance
    why = " They were stripped before reaching the model."
    if boolean_locations:
        why += (
            f" {ALLOW_ENV_VAR} is set, so the boolean toggle under "
            f"{', '.join(sorted(set(boolean_locations)))} was downgraded to this warning "
            "instead of a hard error."
        )
    else:
        why += (
            f" A boolean toggle here would be a hard error; set {ALLOW_ENV_VAR}=1 to "
            "downgrade any such failure to a warning."
        )
    return base + why + guidance


def enforce_text_completions_body(path: str, body: dict[str, Any], *, allow: bool | None = None) -> None:
    """Enforce the chat-template policy on a text-completions request *body*.

    Mutates *body* in place (strips inert fields). No-op for chat endpoints or
    clean bodies. Raises ``ValueError`` when a boolean toggle is present inside
    ``chat_template_kwargs``, unless ``NEMO_EVALUATOR_ALLOW_COMPLETIONS_CHAT_TEMPLATE``
    downgrades it to a warning.
    """
    if not is_text_completions_path(path):
        return
    locations, boolean_locations = _scan_chat_template_fields(body)
    if not locations:
        return
    allow = allow_completions_chat_template() if allow is None else allow
    fatal = bool(boolean_locations) and not allow
    message = _build_message(
        locations, context=f"a request to {path!r}", boolean_locations=boolean_locations, fatal=fatal
    )
    if fatal:
        raise ValueError(message)
    logger.warning(message)
    stripped = _strip_chat_template_fields(body)
    body.clear()
    body.update(stripped)


def normalize_adapter_config_for_endpoint(
    adapter_config: dict[str, Any] | None,
    endpoint_type: str | None,
    *,
    allow: bool | None = None,
) -> dict[str, Any] | None:
    """Enforce the chat-template policy on a legacy ``adapter_config``.

    Only acts when *endpoint_type* selects the text completions endpoint and
    the config carries a chat-template-only field (e.g.
    ``params_to_add.chat_template_kwargs``). Returns the config with inert
    fields stripped, or the input unchanged when there is nothing to do. Raises
    ``ValueError`` when a boolean toggle is present inside ``chat_template_kwargs``,
    unless ``NEMO_EVALUATOR_ALLOW_COMPLETIONS_CHAT_TEMPLATE`` downgrades it to a
    warning.
    """
    if not adapter_config:
        return adapter_config
    if not is_text_completions_endpoint_type(endpoint_type):
        return adapter_config
    locations, boolean_locations = _scan_chat_template_fields(adapter_config)
    if not locations:
        return adapter_config
    allow = allow_completions_chat_template() if allow is None else allow
    fatal = bool(boolean_locations) and not allow
    message = _build_message(
        locations,
        context="adapter_config for a completions endpoint",
        boolean_locations=boolean_locations,
        fatal=fatal,
    )
    if fatal:
        raise ValueError(message)
    logger.warning(message)
    return _strip_chat_template_fields(adapter_config)
