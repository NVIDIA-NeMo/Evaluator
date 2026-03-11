"""Environment registry: discover and resolve by name or URI.

All environments use ``scheme://task`` URI syntax:
  - ``gym://host:port``   (connect to a running Gym server)
  - ``gym://swebench``    (benchmark name -- lifecycle managed by services config)
  - ``skills://mmlu-pro``
  - ``lm-eval://aime2025``
  - ``pi://simpleqa``
  - ``mteb://STSBenchmark``
  - ``container://image#task``

Built-in benchmarks use bare names: ``mmlu``, ``gsm8k``, etc.
"""

from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from nemo_evaluator.environments.base import EvalEnvironment

logger = logging.getLogger(__name__)

_REGISTRY: dict[str, type[EvalEnvironment]] = {}
_builtins_loaded = False
_loaded_files: set[str] = set()

_HOST_PORT_RE = re.compile(r"^[\w.\-]+:\d+$")


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


def load_benchmark_file(file_path: str) -> list[str]:
    """Import an external .py file so its @benchmark/@register decorators fire.

    Returns the list of benchmark names that were newly registered by the file.
    Safe to call multiple times with the same path (idempotent).
    """
    resolved = str(Path(file_path).resolve())
    if resolved in _loaded_files:
        return []

    if not Path(resolved).is_file():
        raise FileNotFoundError(f"Benchmark file not found: {file_path}")

    before_keys = dict(_REGISTRY)
    module_name = f"_nel_ext_{Path(resolved).stem}_{id(resolved)}"
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from: {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    _loaded_files.add(resolved)

    new_names: list[str] = []
    for name in sorted(_REGISTRY):
        if name not in before_keys:
            new_names.append(name)
        elif _REGISTRY[name] is not before_keys[name]:
            logger.warning("External file %s shadows built-in benchmark %r", file_path, name)
            new_names.append(name)

    if new_names:
        logger.info("Loaded benchmarks from %s: %s", file_path, ", ".join(new_names))
    else:
        logger.warning("File %s was imported but registered no benchmarks", file_path)
    return new_names


# --- URI scheme factories (scheme://rest) ---

def _make_gym(rest: str, **kwargs: Any) -> "EvalEnvironment":
    """Auto-detect gym://host:port (remote) vs gym://name (managed benchmark)."""
    from nemo_evaluator.environments.gym import GymEnvironment, ManagedGymEnvironment

    if _HOST_PORT_RE.match(rest):
        return GymEnvironment(f"http://{rest}")
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


def _make_lm_eval(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.lm_eval import LMEvalEnvironment
    limit = kwargs.get("limit") or kwargs.get("num_examples")
    return LMEvalEnvironment(task_name=rest, num_fewshot=kwargs.get("num_fewshot"), limit=limit)


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
    "skills": _make_skills,
    "lm-eval": _make_lm_eval,
    "pi": _make_pi,
    "mteb": _make_mteb,
    "container": _make_container,
}


def _resolve_uri(uri: str, **kwargs: Any) -> "EvalEnvironment | None":
    for scheme, factory in _URI_FACTORIES.items():
        prefix = f"{scheme}://"
        if uri.startswith(prefix):
            return factory(uri[len(prefix):], **kwargs)
    return None


def _resolve_file_bench(name: str) -> str | None:
    """If *name* points to a .py file, import it and return the benchmark name.

    Accepts ``path/to/file.py`` (auto-detect) or ``path/to/file.py:bench_name``.
    Returns None if *name* is not a file reference.
    """
    explicit_name: str | None = None
    file_path = name
    if ":" in name and not name.startswith(("http://", "https://")):
        candidate, explicit_name = name.rsplit(":", 1)
        if candidate.endswith(".py"):
            file_path = candidate

    if not file_path.endswith(".py"):
        return None

    before = set(_REGISTRY)
    new_names = load_benchmark_file(file_path)

    if explicit_name:
        if explicit_name.lower() not in _REGISTRY:
            raise KeyError(f"Benchmark {explicit_name!r} not found after loading {file_path}. "
                           f"Registered: {', '.join(new_names) or '(none)'}")
        return explicit_name

    if len(new_names) == 1:
        return new_names[0]
    if len(new_names) > 1:
        raise KeyError(f"{file_path} registered {len(new_names)} benchmarks "
                       f"({', '.join(new_names)}). Specify one: {file_path}:<name>")

    # File was already loaded on a previous call -- find what it registered
    after = set(_REGISTRY)
    if after == before:
        stem = Path(file_path).stem.lower().replace("-", "_")
        if stem in _REGISTRY:
            return stem

    raise KeyError(f"{file_path} registered no new benchmarks. "
                   f"Make sure it uses @benchmark + @scorer.")


def get_environment(name: str, **kwargs: Any) -> "EvalEnvironment":
    """Resolve an environment by name, URI, or .py file path.

    Resolution order:
      1. File path (``./my_bench.py`` or ``./my_bench.py:name``)
      2. URI scheme (``lm-eval://``, ``skills://``, ``gym://``, etc.)
      3. Built-in registry (``@benchmark`` / ``@register``)
    """
    _ensure_builtins()

    file_name = _resolve_file_bench(name)
    if file_name is not None:
        name = file_name

    uri_env = _resolve_uri(name, **kwargs)
    if uri_env is not None:
        return uri_env

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
