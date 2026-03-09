"""Environment registry: discover and resolve by name or URI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import EvalEnvironment

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[EvalEnvironment]] = {}
_builtins_loaded = False


def _ensure_builtins() -> None:
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        import nemo_evaluator.benchmarks  # noqa: F401


def register(name: str):
    def wrapper(cls: type["EvalEnvironment"]):
        _REGISTRY[name.lower()] = cls
        cls.name = name
        return cls
    return wrapper


# --- Namespace factories (prefix/task) ---

def _make_lm_eval(task: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.lm_eval import LMEvalEnvironment
    limit = kwargs.get("limit") or kwargs.get("num_examples")
    return LMEvalEnvironment(task_name=task, num_fewshot=kwargs.get("num_fewshot"), limit=limit)


_NAMESPACE_FACTORIES: dict[str, Callable[..., "EvalEnvironment"]] = {
    "lm-eval": _make_lm_eval,
}


# --- URI scheme factories (scheme://rest) ---

def _make_gym(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.gym import GymEnvironment
    return GymEnvironment(f"http://{rest}")


def _make_gym_managed(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.gym import ManagedGymEnvironment
    if rest.startswith("cmd:"):
        return ManagedGymEnvironment(server_cmd=rest[4:])
    if rest.startswith("module:"):
        return ManagedGymEnvironment(server_module=rest[7:])
    return ManagedGymEnvironment(nel_benchmark=rest)


def _make_skills(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.skills import SkillsEnvironment
    return SkillsEnvironment(
        rest,
        split=kwargs.get("split"),
        data_dir=kwargs.get("data_dir"),
        prompt_template=kwargs.get("prompt_template"),
        eval_type=kwargs.get("eval_type"),
    )


def _make_pi(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.pi import PIEnvironment
    return PIEnvironment(rest)


def _make_mteb(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.mteb import MTEBEnvironment
    return MTEBEnvironment(task_name=rest)


def _make_container(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.container import ContainerEnvironment
    if "#" in rest:
        image, task = rest.rsplit("#", 1)
    else:
        image, task = rest, ""
    return ContainerEnvironment(image=image, task=task)


_URI_FACTORIES: dict[str, Callable[..., "EvalEnvironment"]] = {
    "gym": _make_gym,
    "gym-managed": _make_gym_managed,
    "skills": _make_skills,
    "pi": _make_pi,
    "mteb": _make_mteb,
    "container": _make_container,
}


def _resolve_namespace(name: str, **kwargs: Any) -> "EvalEnvironment | None":
    if "/" not in name:
        return None
    prefix, task = name.split("/", 1)
    factory = _NAMESPACE_FACTORIES.get(prefix.lower())
    return factory(task, **kwargs) if factory else None


def _resolve_uri(uri: str, **kwargs: Any) -> "EvalEnvironment | None":
    for scheme, factory in _URI_FACTORIES.items():
        prefix = f"{scheme}://"
        if uri.startswith(prefix):
            return factory(uri[len(prefix):], **kwargs)
    return None


def get_environment(name: str, **kwargs: Any) -> "EvalEnvironment":
    """Resolve an environment by name, namespace, or URI.

    Resolution order:
      1. URI scheme (gym://, skills://, pi://, gym-managed://)
      2. Namespace prefix (lm-eval/task)
      3. BYOB registry lookup
    """
    _ensure_builtins()

    uri_env = _resolve_uri(name, **kwargs)
    if uri_env is not None:
        return uri_env

    ns_env = _resolve_namespace(name, **kwargs)
    if ns_env is not None:
        return ns_env

    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise KeyError(f"Unknown environment {name!r}. Available: {available}")
    cls = _REGISTRY[key]
    init_kwargs = {}
    if "num_examples" in kwargs and kwargs["num_examples"] is not None:
        init_kwargs["num_examples"] = kwargs["num_examples"]
    return cls(**init_kwargs)


def list_environments() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)
