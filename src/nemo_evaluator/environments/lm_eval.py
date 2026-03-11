"""lm-eval generate_until tasks as EvalEnvironments."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)

_REWARD_METRIC_KEYS = (
    "exact_match", "acc", "em", "score", "bleu", "rouge1",
    "prompt_level_strict_acc", "prompt_level_loose_acc",
)


class UnsupportedTaskTypeError(RuntimeError):
    """Task requires loglikelihood which chat APIs cannot provide."""


def _ensure_importable() -> None:
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        raise ImportError("lm-evaluation-harness not found. Install: pip install lm_eval")


class LMEvalEnvironment(EvalEnvironment):
    """Wraps an lm-eval generate_until task."""

    def __init__(self, task_name: str, num_fewshot: int | None = None, limit: int | None = None) -> None:
        super().__init__()
        _ensure_importable()
        from lm_eval.tasks import TaskManager, get_task_dict

        self.task_name = task_name
        self.name = f"lm-eval://{task_name}"

        tm = TaskManager()
        task_dict = get_task_dict([task_name], tm)
        self._task = list(task_dict.values())[0]

        output_type = getattr(self._task, "OUTPUT_TYPE",
                              getattr(self._task, "output_type", None))
        if output_type and output_type != "generate_until":
            raise UnsupportedTaskTypeError(
                f"Task {task_name!r} requires {output_type!r}. Only generate_until supported."
            )

        if num_fewshot is not None and hasattr(self._task, "config"):
            self._task.config.num_fewshot = num_fewshot
        if hasattr(self._task, "download"):
            self._task.download()

        self._docs: list[dict] = []
        self._prompts: list[str] = []
        self._gen_kwargs: list[dict] = []
        self._expected: list[str] = []
        self._build_items(limit)
        self._dataset = self._docs
        logger.info("Loaded lm-eval://%s: %d items", task_name, len(self._docs))

    def _build_items(self, limit: int | None) -> None:
        fewshot = 0
        if hasattr(self._task, "config") and self._task.config.num_fewshot is not None:
            fewshot = self._task.config.num_fewshot

        for doc_id, doc in self._task.doc_iterator(rank=0, limit=limit, world_size=1):
            try:
                ctx = self._task.fewshot_context(doc, num_fewshot=fewshot)
            except (TypeError, ValueError, KeyError):
                ctx = ""
            try:
                inst = self._task.construct_requests(doc=doc, ctx=ctx,
                                                     metadata=(self.task_name, doc_id, 1))
                if isinstance(inst, list):
                    inst = inst[0]
                prompt = inst.args[0] if hasattr(inst, "args") else str(ctx)
                gen_kwargs = inst.args[1] if hasattr(inst, "args") and len(inst.args) > 1 else {}
            except (TypeError, ValueError, KeyError, IndexError, AttributeError):
                prompt = str(ctx)
                gen_kwargs = {}
            target = ""
            try:
                target = str(self._task.doc_to_target(doc))
            except (TypeError, ValueError, KeyError):
                pass
            self._docs.append(doc)
            self._prompts.append(prompt)
            self._gen_kwargs.append(gen_kwargs)
            self._expected.append(target.strip())

    def __len__(self) -> int:
        return len(self._docs)

    async def seed(self, idx: int) -> SeedResult:
        return SeedResult(
            prompt=self._prompts[idx],
            expected_answer=self._expected[idx],
            messages=[{"role": "user", "content": self._prompts[idx]}],
            metadata={"_idx": idx, "_task": self.task_name, "_gen_kwargs": self._gen_kwargs[idx]},
        )

    async def verify(self, response: str, expected: str,
                     sandbox: Sandbox | None = None, **meta: Any) -> VerifyResult:
        idx = meta.get("_idx", 0)
        doc = self._docs[idx] if idx < len(self._docs) else {}
        gen_kwargs = meta.get("_gen_kwargs", {})

        text = response
        for stop in gen_kwargs.get("until", []):
            if stop in text:
                text = text[:text.index(stop)]

        try:
            metrics = self._task.process_results(doc, [text])
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
            reward=reward, extracted_answer=text[:500],
            scoring_details={"method": f"lm-eval://{self.task_name}", "metrics": metrics},
        )
