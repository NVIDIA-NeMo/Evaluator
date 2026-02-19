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

    from nemo_evaluator.byob import benchmark, scorer
    from nemo_evaluator.byob.scorers import exact_match

    @benchmark(
        name="my-qa",
        dataset="/path/to/data.jsonl",
        prompt="Question: {question}\\nAnswer:",
        target_field="answer",
    )
    @scorer
    def my_scorer(response, target, metadata):
        return exact_match(response, target, metadata)
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict


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


def benchmark(name: str, dataset: str, prompt: str,
              target_field: str = "target", endpoint_type: str = "chat", **kwargs):
    """Decorator that registers a function as a BYOB benchmark.

    Args:
        name: Human-readable benchmark name.
        dataset: Path to JSONL dataset file.
        prompt: Python format string with {field} placeholders.
        target_field: JSONL field containing ground truth.
        endpoint_type: "chat" or "completions".
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

        defn = BenchmarkDefinition(
            name=name,
            normalized_name=normalized,
            dataset=dataset,
            prompt=prompt,
            scorer_fn=fn,
            target_field=target_field,
            endpoint_type=endpoint_type,
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
