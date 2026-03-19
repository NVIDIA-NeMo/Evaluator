"""Interceptor registry for the LiteLLM proxy.

Custom interceptors are subclasses of :class:`BaseInterceptor` (which extends
``litellm.integrations.custom_logger.CustomLogger``).  They live in this
package and are registered by name so the YAML config can reference them.

Names that don't match a registered custom interceptor are passed through as
built-in LiteLLM callback names (e.g. ``"langfuse"``, ``"prometheus"``).
"""
from __future__ import annotations

import importlib
import logging
from typing import Any

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, str] = {
    "log_tokens": "nemo_evaluator.interceptors.log_tokens",
}
"""Maps interceptor short-names to their fully-qualified module paths.
Each module must expose an ``Interceptor`` class attribute."""


def register(name: str, module_path: str) -> None:
    """Register a custom interceptor by name."""
    _REGISTRY[name] = module_path


def resolve_interceptors(names: list[str]) -> list[Any]:
    """Resolve interceptor names to callback objects.

    Custom interceptors (registered in this package) are instantiated.
    Unknown names are returned as plain strings — LiteLLM treats those as
    built-in callback names (e.g. ``"langfuse"``).
    """
    result: list[Any] = []
    for name in names:
        if name in _REGISTRY:
            mod = importlib.import_module(_REGISTRY[name])
            cls = getattr(mod, "Interceptor", None)
            if cls is None:
                raise AttributeError(
                    f"Interceptor module {_REGISTRY[name]!r} has no 'Interceptor' class"
                )
            result.append(cls())
            logger.info("Loaded custom interceptor: %s", name)
        else:
            result.append(name)
            logger.info("Using built-in LiteLLM callback: %s", name)
    return result
