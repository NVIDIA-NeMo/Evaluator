"""Adapters: Gym harness server, proxy server, container runner."""
from nemo_evaluator.adapters.gym_harness import GymHarness

__all__ = ["GymHarness", "ContainerConfig", "run_container_eval"]

_LAZY = {
    "ContainerConfig": ("nemo_evaluator.adapters.container", "ContainerConfig"),
    "run_container_eval": ("nemo_evaluator.adapters.container", "run_container_eval"),
}


def __getattr__(name: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
