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
"""lm-eval generate_until tasks as EvalEnvironments.

Supports both single tasks (e.g. ``lm-eval://gsm8k``) and task groups
(e.g. ``lm-eval://mmlu``).  Groups are recursively flattened into their
leaf subtasks, and each item tracks its originating subtask for correct
scoring dispatch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_REWARD_METRIC_KEYS = (
    "exact_match",
    "acc",
    "em",
    "score",
    "bleu",
    "rouge1",
    "prompt_level_strict_acc",
    "prompt_level_loose_acc",
)


class UnsupportedTaskTypeError(RuntimeError):
    """Task requires loglikelihood which chat APIs cannot provide."""


def _ensure_importable() -> None:
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        raise ImportError("lm-evaluation-harness not found. Install: pip install lm_eval")


def _collect_leaf_tasks(obj: Any) -> list[Any]:
    """Recursively extract leaf task objects from a possibly nested dict."""
    if isinstance(obj, dict):
        leaves: list[Any] = []
        for v in obj.values():
            leaves.extend(_collect_leaf_tasks(v))
        return leaves
    return [obj]


class LMEvalEnvironment(EvalEnvironment):
    """Wraps one or more lm-eval generate_until tasks (including groups)."""

    def __init__(self, task_name: str, num_fewshot: int | None = None, limit: int | None = None) -> None:
        super().__init__()
        _ensure_importable()
        from lm_eval.tasks import TaskManager, get_task_dict

        self.task_name = task_name
        self.name = f"lm-eval://{task_name}"

        tm = TaskManager()
        task_dict = get_task_dict([task_name], tm)
        self._tasks = _collect_leaf_tasks(task_dict)
        if not self._tasks:
            raise ValueError(f"No leaf tasks found for {task_name!r}")

        kept: list[Any] = []
        for t in self._tasks:
            ot = getattr(t, "OUTPUT_TYPE", getattr(t, "output_type", None))
            if ot and ot != "generate_until":
                tname = getattr(t, "task_name", getattr(getattr(t, "config", None), "task", "?"))
                logger.warning("Skipping subtask %s (output_type=%s)", tname, ot)
                continue
            if num_fewshot is not None and hasattr(t, "config"):
                t.config.num_fewshot = num_fewshot
            if hasattr(t, "download"):
                t.download()
            kept.append(t)
        if not kept:
            raise UnsupportedTaskTypeError(
                f"All subtasks of {task_name!r} require loglikelihood. Only generate_until supported."
            )
        self._tasks = kept

        self._docs: list[dict] = []
        self._prompts: list[str] = []
        self._gen_kwargs: list[dict] = []
        self._expected: list[str] = []
        self._task_idx: list[int] = []
        self._build_items(limit)
        self._dataset = self._docs
        logger.info("Loaded lm-eval://%s: %d items across %d subtask(s)", task_name, len(self._docs), len(self._tasks))

    def _build_items(self, limit: int | None) -> None:
        per_task_limit = limit
        for ti, task in enumerate(self._tasks):
            fewshot = 0
            if hasattr(task, "config") and task.config.num_fewshot is not None:
                fewshot = task.config.num_fewshot

            task_name = getattr(task, "task_name", getattr(getattr(task, "config", None), "task", self.task_name))

            for doc_id, doc in task.doc_iterator(rank=0, limit=per_task_limit, world_size=1):
                try:
                    ctx = task.fewshot_context(doc, num_fewshot=fewshot)
                except (TypeError, ValueError, KeyError):
                    ctx = ""
                try:
                    inst = task.construct_requests(doc=doc, ctx=ctx, metadata=(task_name, doc_id, 1))
                    if isinstance(inst, list):
                        inst = inst[0]
                    prompt = inst.args[0] if hasattr(inst, "args") else str(ctx)
                    gen_kwargs = inst.args[1] if hasattr(inst, "args") and len(inst.args) > 1 else {}
                except (TypeError, ValueError, KeyError, IndexError, AttributeError):
                    prompt = str(ctx)
                    gen_kwargs = {}
                target = ""
                try:
                    target = str(task.doc_to_target(doc))
                except (TypeError, ValueError, KeyError):
                    pass
                self._docs.append(doc)
                self._prompts.append(prompt)
                self._gen_kwargs.append(gen_kwargs)
                self._expected.append(target.strip())
                self._task_idx.append(ti)

    def __len__(self) -> int:
        return len(self._docs)

    async def seed(self, idx: int) -> SeedResult:
        return SeedResult(
            prompt=self._prompts[idx],
            expected_answer=self._expected[idx],
            messages=[{"role": "user", "content": self._prompts[idx]}],
            metadata={"_idx": idx, "_task": self.task_name, "_gen_kwargs": self._gen_kwargs[idx]},
        )

    async def verify(self, response: str, expected: str, sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        idx = meta.get("_idx", 0)
        doc = self._docs[idx] if idx < len(self._docs) else {}
        gen_kwargs = meta.get("_gen_kwargs", {})
        task = self._tasks[self._task_idx[idx]] if idx < len(self._task_idx) else self._tasks[0]

        text = response
        for stop in gen_kwargs.get("until", []):
            if stop in text:
                text = text[: text.index(stop)]

        try:
            metrics = task.process_results(doc, [text])
        except (TypeError, ValueError, KeyError) as exc:
            logger.warning("process_results failed for %s doc %d: %s", self.task_name, idx, exc)
            metrics = {}

        reward = 0.0
        if isinstance(metrics, dict):
            for key in _REWARD_METRIC_KEYS:
                if key in metrics:
                    val = metrics[key]
                    if isinstance(val, (list, tuple)):
                        val = sum(val) / len(val) if val else 0.0
                    reward = float(val)
                    break

        return VerifyResult(
            reward=reward,
            extracted_answer=text[:500],
            scoring_details={"method": f"lm-eval://{self.task_name}", "lm_eval_metrics": metrics},
        )
