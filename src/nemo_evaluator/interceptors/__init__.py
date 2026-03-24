"""Interceptor registry for the LiteLLM proxy.

Custom interceptors are subclasses of :class:`BaseInterceptor` (which extends
``litellm.integrations.custom_logger.CustomLogger``).  They live in this
package and are registered by name so the YAML config can reference them.

Names that don't match a registered custom interceptor are passed through as
built-in LiteLLM callback names (e.g. ``"langfuse"``, ``"prometheus"``).

Entries in the ``interceptors`` list can be plain strings (no config) or
single-key dicts whose value is forwarded as ``**kwargs`` to the
interceptor constructor::

    interceptors:
      - log_tokens
      - modify_tools:
          strip_properties: [security_risk]
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "log_tokens": "nemo_evaluator.interceptors.log_tokens",
    "modify_tools": "nemo_evaluator.interceptors.modify_tools",
    "turn_counter": "nemo_evaluator.interceptors.turn_counter",
}
"""Maps interceptor short-names to their fully-qualified module paths.
Each module must expose an ``Interceptor`` class attribute."""


def register(name: str, module_path: str) -> None:
    """Register a custom interceptor by name."""
    _REGISTRY[name] = module_path


def _parse_entry(entry: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(name, kwargs)`` from a plain string or single-key dict."""
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and len(entry) == 1:
        name = next(iter(entry))
        cfg = entry[name]
        return name, cfg if isinstance(cfg, dict) else {}
    raise ValueError(f"Interceptor entry must be a string or a single-key dict, got: {entry!r}")


def resolve_interceptors(entries: list[str | dict[str, Any]]) -> list[Any]:
    """Resolve interceptor entries to callback objects.

    Custom interceptors (registered in this package) are instantiated with
    any config kwargs from the YAML.  Unknown names are returned as plain
    strings — LiteLLM treats those as built-in callback names.
    """
    result: list[Any] = []
    for entry in entries:
        name, kwargs = _parse_entry(entry)
        if name in _REGISTRY:
            mod = importlib.import_module(_REGISTRY[name])
            cls = getattr(mod, "Interceptor", None)
            if cls is None:
                raise AttributeError(f"Interceptor module {_REGISTRY[name]!r} has no 'Interceptor' class")
            result.append(cls(**kwargs))
            logger.info("Loaded custom interceptor: %s (config=%s)", name, kwargs or "-")
        else:
            result.append(name)
            logger.info("Using built-in LiteLLM callback: %s", name)
    return result
