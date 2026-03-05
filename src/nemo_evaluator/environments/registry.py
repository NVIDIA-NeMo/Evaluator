"""Discover and register EvalEnvironment subclasses by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import EvalEnvironment


_REGISTRY: dict[str, type[EvalEnvironment]] = {}
_builtins_loaded = False


def _ensure_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        import nemo_evaluator.benchmarks  # noqa: F401


def register(name: str):
    """Class decorator that registers an EvalEnvironment subclass under *name*."""

    def wrapper(cls: type[EvalEnvironment]):
        _REGISTRY[name.lower()] = cls
        cls.name = name
        return cls

    return wrapper


def get_environment(name: str) -> EvalEnvironment:
    """Instantiate and return an EvalEnvironment by registered name."""
    _ensure_builtins()
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown benchmark {name!r}. Available: {available}")
    return _REGISTRY[key]()


def list_environments() -> list[str]:
    """Return all registered benchmark names."""
    _ensure_builtins()
    return sorted(_REGISTRY)
