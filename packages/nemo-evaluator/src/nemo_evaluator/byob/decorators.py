# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""BYOB decorators and benchmark registry.

The @benchmark and @scorer decorators are the user-facing API for defining
custom evaluation benchmarks.

Example::

    from nemo_evaluator.byob import benchmark, scorer, ScorerInput
    from nemo_evaluator.byob.scorers import exact_match

    @benchmark(
        name="my-qa",
        dataset="/path/to/data.jsonl",
        prompt="Question: {question}\\nAnswer:",
        target_field="answer",
    )
    @scorer
    def my_scorer(inp: ScorerInput):
        return exact_match(inp.response, inp.target, inp.metadata)
"""

import inspect
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ScorerInput:
    """All inputs available to a scorer function.

    This is the single argument passed to all BYOB scorer functions.
    Standard scorers use response, target, and metadata.
    Advanced scorers (judge, multi-turn) use the optional fields.
    """
    response: str
    target: str
    metadata: dict
    # Extensible fields for advanced scorers
    model_call_fn: Optional[Callable] = None
    config: Dict[str, Any] = field(default_factory=dict)
    conversation: Optional[List[dict]] = None
    turn_index: Optional[int] = None


@dataclass
class BenchmarkDefinition:
    """Internal representation of a registered BYOB benchmark."""

    name: str
    normalized_name: str
    dataset: str
    prompt: str
    scorer_fn: Callable
    target_field: str = "target"
    endpoint_type: str = "chat"
    requirements: List[str] = field(default_factory=list)
    extra_config: Dict[str, Any] = field(default_factory=dict)


_BENCHMARK_REGISTRY: Dict[str, BenchmarkDefinition] = {}


def _normalize_name(name: str) -> str:
    """Normalize a benchmark name to a valid Python identifier.

    Rules:
    - Lowercase
    - Replace non-alphanumeric characters with underscores
    - Collapse consecutive underscores
    - Strip leading/trailing underscores
    - Truncate to 50 characters
    """
    normalized = re.sub(r"[^a-z0-9]", "_", name.lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized[:50]


def _resolve_prompt(prompt: str, base_dir: Path) -> str:
    """If prompt ends with a template file extension, read it as a file. Otherwise return as-is."""
    if any(prompt.endswith(ext) for ext in (".txt", ".md", ".jinja", ".jinja2")):
        path = (base_dir / prompt) if not Path(prompt).is_absolute() else Path(prompt)
        if path.exists():
            return path.read_text(encoding="utf-8")
    return prompt


def _resolve_requirements(requirements, base_dir: Path) -> List[str]:
    """Resolve requirements: str means file path, list means inline deps, None means no deps."""
    if requirements is None:
        return []
    if isinstance(requirements, str):
        path = (base_dir / requirements) if not Path(requirements).is_absolute() else Path(requirements)
        if not path.exists():
            raise FileNotFoundError(f"Requirements file not found: {path}")
        return [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    return list(requirements)


def benchmark(name: str, dataset: str, prompt: str,
              target_field: str = "target", endpoint_type: str = "chat",
              requirements=None, **kwargs):
    """Decorator that registers a function as a BYOB benchmark.

    Args:
        name: Human-readable benchmark name.
        dataset: Path to JSONL dataset file.
        prompt: Python format string with {field} placeholders, or path to
                a template file (.txt, .md, .jinja, .jinja2).
        target_field: JSONL field containing ground truth.
        endpoint_type: "chat" or "completions".
        requirements: Pip dependencies. Either a list of specifiers
                      (e.g., ["rouge-score>=0.1.2"]) or a path to a
                      requirements.txt file. None means no extra deps.
        **kwargs: Extra configuration passed through to extra_config.
    """
    def decorator(fn):
        normalized = _normalize_name(name)
        if not normalized:
            raise ValueError(
                f"Benchmark name '{name}' normalizes to an empty string. "
                f"Use a name containing at least one alphanumeric character."
            )
        if normalized in _BENCHMARK_REGISTRY:
            raise ValueError(
                f"Benchmark '{name}' (normalized: '{normalized}') is already registered."
            )

        # Resolve base_dir from decorated function's source file
        try:
            source_file = inspect.getfile(fn)
            base_dir = Path(source_file).parent
        except (TypeError, OSError):
            base_dir = Path.cwd()

        resolved_prompt = _resolve_prompt(prompt, base_dir)
        resolved_reqs = _resolve_requirements(requirements, base_dir)

        defn = BenchmarkDefinition(
            name=name,
            normalized_name=normalized,
            dataset=dataset,
            prompt=resolved_prompt,
            scorer_fn=fn,
            target_field=target_field,
            endpoint_type=endpoint_type,
            requirements=resolved_reqs,
            extra_config=kwargs,
        )

        _BENCHMARK_REGISTRY[normalized] = defn
        fn._benchmark_definition = defn
        return fn

    return decorator


def scorer(fn):
    """Decorator that marks a function as a BYOB scorer.

    Sets fn._is_scorer = True. Used compositionally with @benchmark
    (scorer is the inner decorator).
    """
    fn._is_scorer = True
    return fn


def get_registered_benchmarks() -> Dict[str, BenchmarkDefinition]:
    """Return a copy of the benchmark registry."""
    return dict(_BENCHMARK_REGISTRY)


def clear_registry():
    """Clear all registered benchmarks."""
    _BENCHMARK_REGISTRY.clear()
