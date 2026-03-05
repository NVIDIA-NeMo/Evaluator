"""Adapter for Prime Intellect verifiers environments.

Two modes:
  1. Decomposed (SingleTurnEnv): Evaluator owns the model call via seed/verify.
     Full trajectory capture, n_repeats, confidence intervals.
  2. Native (any env): Delegates to env.evaluate() with our client.
     Uses PI's native trajectory/timing data. For MultiTurnEnv/SandboxEnv.
"""

from __future__ import annotations

import logging
from typing import Any

from nemo_evaluator.adapters.base import EnvironmentAdapter
from nemo_evaluator.environments.base import SeedResult, VerifyResult

logger = logging.getLogger(__name__)


def _require_verifiers():
    try:
        import verifiers as vf
        return vf
    except ImportError:
        raise ImportError("PIAdapter requires 'verifiers'. Install: pip install verifiers")


class PIAdapter(EnvironmentAdapter):
    """Decomposes a verifiers SingleTurnEnv into seed/verify.

    Evaluator makes the model call between seed and verify, giving us
    full trajectory data, n_repeats, and confidence intervals.
    """

    def __init__(
        self,
        env_name: str,
        env_kwargs: dict[str, Any] | None = None,
        question_key: str = "question",
        answer_key: str = "answer",
    ) -> None:
        self._env_name = env_name
        self._q_key = question_key
        self._a_key = answer_key
        self.name = f"pi:{env_name}"

        vf = _require_verifiers()
        self._vf_env = vf.load_environment(env_name, **(env_kwargs or {}))
        self._dataset = self._vf_env.get_eval_dataset()
        self._rubric = getattr(self._vf_env, "rubric", None)
        self._system_prompt = getattr(self._vf_env, "system_prompt", None)
        logger.info("PI env %s: %d problems", env_name, len(self._dataset))

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        prompt = str(row.get(self._q_key, row.get("prompt", "")))
        answer = str(row.get(self._a_key, ""))
        meta: dict[str, Any] = {"source": "pi", "env": self._env_name}
        for k in ("task", "category", "difficulty", "info"):
            if k in row:
                meta[k] = row[k]
        return SeedResult(prompt=prompt, expected_answer=answer, metadata=meta)

    async def verify(self, response: str, expected: str, **meta: Any) -> VerifyResult:
        if self._rubric is not None:
            reward, details = await self._score_via_rubric(response, expected, meta)
        else:
            reward = 1.0 if response.strip().lower() == expected.strip().lower() else 0.0
            details = {"method": "exact_match_fallback"}

        return VerifyResult(
            reward=reward,
            extracted_answer=response.strip(),
            scoring_details=details,
        )

    async def _score_via_rubric(self, response: str, expected: str,
                                meta: dict[str, Any]) -> tuple[float, dict]:
        vf = _require_verifiers()
        # Build minimal State for rubric scoring
        state: dict[str, Any] = {
            "input": {
                "prompt": meta.get("_prompt", ""),
                "answer": expected,
                "example_id": meta.get("problem_idx", 0),
                "task": meta.get("task", self._env_name),
            },
            "completion": response,
            "reward": None,
            "metrics": None,
            "timing": {"start_time": 0, "generation_ms": 0, "scoring_ms": 0, "total_ms": 0},
            "trajectory": [],
            "is_completed": True,
            "is_truncated": False,
        }

        try:
            await self._rubric.score_rollout(state)
            return (
                float(state.get("reward", 0.0)),
                {"method": "verifiers_rubric", "metrics": state.get("metrics", {})},
            )
        except Exception as e:
            logger.warning("Rubric scoring failed, falling back: %s", e)
            reward = 1.0 if response.strip().lower() == expected.strip().lower() else 0.0
            return reward, {"method": "exact_match_fallback", "rubric_error": str(e)}

    async def dataset_size(self) -> int:
        return len(self._dataset)

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt


class PINativeRunner:
    """Runs a verifiers env natively via env.evaluate().

    For MultiTurnEnv/SandboxEnv where decomposition isn't possible.
    Captures PI's native trajectory data and converts to our artifact format.
    """

    def __init__(self, env_name: str, env_kwargs: dict[str, Any] | None = None) -> None:
        vf = _require_verifiers()
        self._env = vf.load_environment(env_name, **(env_kwargs or {}))
        self.name = f"pi-native:{env_name}"

    async def run(
        self,
        model: str,
        base_url: str,
        api_key: str,
        num_examples: int = -1,
        rollouts_per_example: int = 1,
        max_concurrent: int = 8,
    ) -> dict[str, Any]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        outputs = await self._env.evaluate(
            client=client,
            model=model,
            num_examples=num_examples,
            rollouts_per_example=rollouts_per_example,
            max_concurrent=max_concurrent,
        )
        return self._convert_outputs(outputs)

    def _convert_outputs(self, outputs: dict) -> dict[str, Any]:
        """Convert PI's GenerateOutputs into our artifact format."""
        results = []
        for group_idx, group in enumerate(outputs.get("groups", [])):
            for rollout_idx, rollout in enumerate(group.get("rollouts", [])):
                state = rollout.get("state", {})
                results.append({
                    "problem_idx": group_idx,
                    "repeat": rollout_idx,
                    "reward": state.get("reward", 0.0),
                    "metrics": state.get("metrics", {}),
                    "timing": state.get("timing", {}),
                    "usage": state.get("usage", {}),
                    "trajectory_steps": len(state.get("trajectory", [])),
                })
        return {"source": "pi_native", "results": results}
