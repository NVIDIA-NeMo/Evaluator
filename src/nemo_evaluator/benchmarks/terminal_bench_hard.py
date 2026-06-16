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
"""Terminal-Bench Hard — curated leaderboard subsets of TB v1.

Registers ``terminal-bench-hard`` (47 tasks) and
``terminal-bench-hard-aa-split`` (44 tasks) as built-in benchmarks.
Both reuse :func:`~nemo_evaluator.benchmarks.terminal_bench_v1._ensure_dataset`
to clone and map the full TB v1 repo, then filter to their task list.

Config usage::

    benchmarks:
      - name: terminal-bench-hard          # 47 tasks
      - name: terminal-bench-hard-aa-split # 44 tasks
"""

from __future__ import annotations

import logging
from pathlib import Path

from nemo_evaluator.benchmarks.terminal_bench_v1 import _ensure_dataset
from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.harbor import HarborEnvironment
from nemo_evaluator.environments.registry import register

logger = logging.getLogger(__name__)

# Curated leaderboard subset — 47 tasks present at HEAD of
# laude-institute/terminal-bench (original-tasks/ directory).
_TB_HARD_TASKS: frozenset[str] = frozenset(
    {
        "aimo-airline-departures",
        "blind-maze-explorer-5x5",
        "cartpole-rl-training",
        "causal-inference-r",
        "chem-property-targeting",
        "chem-rf",
        "circuit-fibsqrt",
        "cobol-modernization",
        "configure-git-webserver",
        "cross-entropy-method",
        "extract-moves-from-video",
        "feal-differential-cryptanalysis",
        "feal-linear-cryptanalysis",
        "form-filling",
        "git-multibranch",
        "gpt2-codegolf",
        "install-windows-3.11",
        "install-windows-xp",
        "lean4-proof",
        "make-doom-for-mips",
        "make-mips-interpreter",
        "mcmc-sampling-stan",
        "model-extraction-relu-logits",
        "movie-helper",
        "neuron-to-jaxley-conversion",
        "oom",
        "organization-json-generator",
        "parallel-particle-simulator",
        "parallelize-graph",
        "password-recovery",
        "path-tracing",
        "path-tracing-reverse",
        "play-zork",
        "play-zork-easy",
        "polyglot-rust-c",
        "prove-plus-comm",
        "pytorch-model-cli",
        "rare-mineral-allocation",
        "recover-obfuscated-files",
        "reverse-engineering",
        "run-pdp11-code",
        "stable-parallel-kmeans",
        "swe-bench-astropy-1",
        "swe-bench-astropy-2",
        "train-fasttext",
        "word2vec-from-scratch",
        "write-compressor",
    }
)


@register("terminal-bench-hard")
class TerminalBenchHard(HarborEnvironment):
    """Terminal-Bench Hard — 47-task leaderboard subset, auto-downloaded."""

    _TASK_SET: frozenset[str] = _TB_HARD_TASKS
    _LABEL: str = "terminal-bench-hard"
    # Tasks that fail even under the oracle solver — infra / task / upstream caps,
    # not model capability. When non-empty, ``seed`` tags each task's ``category``
    # as "solvable"/"unsolvable" so the standard category breakdown surfaces both
    # subset accuracies alongside the full headline. Empty here; subsets opt in.
    _UNSOLVABLE_TASKS: frozenset[str] = frozenset()

    def __init__(self, num_examples: int | None = None) -> None:
        dataset_path = _ensure_dataset()
        super().__init__(dataset_path=dataset_path, num_examples=None)
        self._tasks = self._filter_tasks(num_examples)

    def _filter_tasks(self, num_examples: int | None) -> list[Path]:
        tasks = [t for t in self._tasks if t.name in self._TASK_SET]
        found = {t.name for t in tasks}
        missing = self._TASK_SET - found
        if missing:
            logger.warning(
                "%s: %d/%d expected tasks missing after mapping: %s",
                self._LABEL,
                len(missing),
                len(self._TASK_SET),
                sorted(missing),
            )
        if num_examples is not None:
            tasks = tasks[:num_examples]
        return tasks

    @classmethod
    def _subset_category(cls, task_id: str) -> str | None:
        """Solvable/unsolvable bucket for ``task_id``; None when the benchmark
        declares no unsolvable tasks (leaves ``category`` unset)."""
        if not cls._UNSOLVABLE_TASKS:
            return None
        return "unsolvable" if task_id in cls._UNSOLVABLE_TASKS else "solvable"

    async def seed(self, idx: int) -> SeedResult:
        result = await super().seed(idx)
        category = self._subset_category(result.metadata.get("task_id", ""))
        if category is not None:
            result.metadata["category"] = category
        return result


# AA-split — 44-task subset used by the AA scoring pipeline.
_TB_HARD_AA_SPLIT_TASKS: frozenset[str] = frozenset(
    {
        "aimo-airline-departures",
        "blind-maze-explorer-5x5",
        "cartpole-rl-training",
        "chem-property-targeting",
        "chem-rf",
        "circuit-fibsqrt",
        "cobol-modernization",
        "configure-git-webserver",
        "cross-entropy-method",
        "extract-moves-from-video",
        "feal-differential-cryptanalysis",
        "feal-linear-cryptanalysis",
        "form-filling",
        "git-multibranch",
        "gpt2-codegolf",
        "install-windows-xp",
        "make-doom-for-mips",
        "make-mips-interpreter",
        "model-extraction-relu-logits",
        "movie-helper",
        "neuron-to-jaxley-conversion",
        "oom",
        "organization-json-generator",
        "parallel-particle-simulator",
        "parallelize-graph",
        "password-recovery",
        "path-tracing",
        "path-tracing-reverse",
        "play-zork",
        "play-zork-easy",
        "polyglot-rust-c",
        "prove-plus-comm",
        "pytorch-model-cli",
        "rare-mineral-allocation",
        "recover-obfuscated-files",
        "reverse-engineering",
        "run-pdp11-code",
        "stable-parallel-kmeans",
        "super-benchmark-upet",
        "swe-bench-astropy-1",
        "swe-bench-astropy-2",
        "train-fasttext",
        "word2vec-from-scratch",
        "write-compressor",
    }
)


# AA-split tasks that fail even under the oracle solver — known infra/task/upstream
# caps (oracle KNOWN_FAILURES; see FEVPR-175), not model capability. Tagged as the
# "unsolvable" category so the "solvable"-subset accuracy is reported alongside the
# full headline.
_TB_HARD_AA_SPLIT_UNSOLVABLE: frozenset[str] = frozenset(
    {
        "feal-differential-cryptanalysis",
        "install-windows-xp",
        "make-doom-for-mips",
        "run-pdp11-code",
    }
)


@register("terminal-bench-hard-aa-split")
class TerminalBenchHardAASplit(TerminalBenchHard):
    """Terminal-Bench Hard AA-split — 44-task subset for AA scoring."""

    _TASK_SET: frozenset[str] = _TB_HARD_AA_SPLIT_TASKS
    _LABEL: str = "terminal-bench-hard-aa-split"
    _UNSOLVABLE_TASKS: frozenset[str] = _TB_HARD_AA_SPLIT_UNSOLVABLE
