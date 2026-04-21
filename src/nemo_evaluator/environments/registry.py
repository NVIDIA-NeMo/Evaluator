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
"""Environment registry: resolve benchmarks by name or URI scheme."""

from __future__ import annotations

import importlib.util
import inspect
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
    from nemo_evaluator.environments.gym import GymDataset, GymEnvironment, ManagedGymEnvironment

    protocol = kwargs.get("protocol", "evaluator")
    data_path = kwargs.get("data")
    dataset = GymDataset(data_path) if data_path else None

    # Parse inline query params: gym://host:port?protocol=native&data=/foo.jsonl
    if "?" in rest:
        rest_base, qs = rest.split("?", 1)
        for kv in qs.split("&"):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            if k == "protocol":
                protocol = v
            elif k == "data":
                dataset = GymDataset(v)
        rest = rest_base

    if _HOST_PORT_RE.match(rest):
        return GymEnvironment(f"http://{rest}", protocol=protocol, dataset=dataset)
    if rest.startswith("cmd:"):
        return ManagedGymEnvironment(server_cmd=rest[4:], protocol=protocol, dataset=dataset)
    if rest.startswith("module:"):
        return ManagedGymEnvironment(server_module=rest[7:], protocol=protocol, dataset=dataset)
    return ManagedGymEnvironment(nel_benchmark=rest, protocol=protocol, dataset=dataset)


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


def _make_vlmevalkit(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.vlmevalkit import VLMEvalKitEnvironment

    limit = kwargs.get("limit") or kwargs.get("num_examples")
    return VLMEvalKitEnvironment(dataset_name=rest, limit=limit)


def _make_container(rest: str, **kwargs: Any) -> "EvalEnvironment":
    from nemo_evaluator.environments.container import ContainerEnvironment

    if "#" in rest:
        image, task = rest.rsplit("#", 1)
    else:
        image, task = rest, ""
    return ContainerEnvironment(image=image, task=task)


def _make_harbor(rest: str, **kwargs: Any) -> "EvalEnvironment":
    import os

    from nemo_evaluator.environments.harbor import HarborEnvironment, auto_prepare

    datasets_dir = os.environ.get("HARBOR_DATASETS_DIR", "./harbor_datasets")
    safe_dir_name = rest.replace("/", "__")
    dataset_path = Path(datasets_dir) / safe_dir_name

    limit = kwargs.get("num_examples")
    if not auto_prepare(rest, dataset_path, limit=limit):
        if not dataset_path.is_dir():
            raise KeyError(f"Harbor dataset {rest!r} not found in registry and {dataset_path} does not exist on disk.")

    return HarborEnvironment(
        dataset_path=dataset_path,
        num_examples=kwargs.get("num_examples"),
    )


_URI_FACTORIES: dict[str, Callable[..., "EvalEnvironment"]] = {
    "gym": _make_gym,
    "skills": _make_skills,
    "lm-eval": _make_lm_eval,
    "vlmevalkit": _make_vlmevalkit,
    "container": _make_container,
    "harbor": _make_harbor,
}


def _resolve_uri(uri: str, **kwargs: Any) -> "EvalEnvironment | None":
    for scheme, factory in _URI_FACTORIES.items():
        prefix = f"{scheme}://"
        if uri.startswith(prefix):
            return factory(uri[len(prefix) :], **kwargs)
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
            raise KeyError(
                f"Benchmark {explicit_name!r} not found after loading {file_path}. "
                f"Registered: {', '.join(new_names) or '(none)'}"
            )
        return explicit_name

    if len(new_names) == 1:
        return new_names[0]
    if len(new_names) > 1:
        raise KeyError(
            f"{file_path} registered {len(new_names)} benchmarks "
            f"({', '.join(new_names)}). Specify one: {file_path}:<name>"
        )

    after = set(_REGISTRY)
    if after == before:
        stem = Path(file_path).stem.lower().replace("-", "_")
        if stem in _REGISTRY:
            return stem

    raise KeyError(f"{file_path} registered no new benchmarks. Make sure it uses @benchmark + @scorer.")


def _filter_init_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs the target class ``__init__`` actually accepts.

    ``num_examples`` / ``num_fewshot`` are passed opportunistically by the
    orchestrator to every environment; silently drop them for classes that
    don't declare them.  Any *explicit* ``params:`` key that the target
    doesn't accept is a user error and raises a clear ``TypeError``.
    """
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return dict(kwargs)

    params = sig.parameters
    accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return dict(kwargs)

    accepted = {name for name in params if name != "self"}
    return {k: v for k, v in kwargs.items() if k in accepted}


def get_environment(name: str, **kwargs: Any) -> "EvalEnvironment":
    """Resolve an environment by name, URI, or .py file path.

    ``**kwargs`` are forwarded to the target environment constructor, filtered
    to only the keys that constructor declares.  Callers should pass any YAML
    ``params:`` entries alongside the usual ``num_examples`` / ``num_fewshot``
    hints -- unknown ``params:`` keys surface as a ``TypeError`` from Python's
    usual argument binding.
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

    user_params: dict[str, Any] = kwargs.pop("params", None) or {}
    auto_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    init_kwargs = _filter_init_kwargs(cls, auto_kwargs)
    init_kwargs.update(user_params)
    try:
        return cls(**init_kwargs)
    except TypeError as exc:
        raise TypeError(f"Could not instantiate benchmark {name!r} with params {sorted(user_params)}: {exc}") from exc


def list_environments() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)
