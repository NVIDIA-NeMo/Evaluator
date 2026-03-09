"""MTEB environment: wraps the MTEB library for embedding evaluation.

MTEB owns its internal eval loop. This environment delegates to it and
maps results back into the NEL artifact bundle format.

Usage in config:
    benchmarks:
      - name: mteb://MTEB/STSBenchmark
        endpoint_type: embedding
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox
from typing import Any

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

logger = logging.getLogger(__name__)


class MTEBEnvironment(EvalEnvironment):
    """Runs an MTEB task using NEL's ModelClient for embeddings.

    Because MTEB owns its own eval loop, this environment implements
    run_batch() instead of the standard seed/verify cycle.
    """

    def __init__(self, task_name: str, languages: list[str] | None = None) -> None:
        super().__init__()
        self.name = f"mteb/{task_name}"
        self._task_name = task_name
        self._languages = languages

    async def dataset_size(self) -> int:
        return 0

    async def seed(self, idx: int) -> SeedResult:
        raise NotImplementedError("MTEBEnvironment uses run_batch()")

    async def verify(self, response: str, expected: str,
                     sandbox: Sandbox | None = None, **metadata: Any) -> VerifyResult:
        raise NotImplementedError("MTEBEnvironment uses run_batch()")

    async def run_batch(self, solver: Any = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
        """Run the MTEB evaluation and return an artifact bundle."""
        try:
            import mteb
        except ImportError:
            raise ImportError("MTEB is required: pip install mteb")

        client = getattr(solver, "_client", None)
        if client is None:
            raise ValueError("MTEBEnvironment requires an EmbeddingSolver with a ModelClient")

        encoder = _NELEncoder(client)

        tasks = mteb.get_tasks(tasks=[self._task_name])
        if not tasks:
            raise ValueError(f"Unknown MTEB task: {self._task_name}")

        evaluation = mteb.MTEB(tasks=tasks)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None, lambda: evaluation.run(encoder, output_folder=None, verbosity=1)
        )

        return _mteb_results_to_bundle(self._task_name, results)


class _NELEncoder:
    """MTEB-compatible encoder backed by ModelClient.embed_batch().

    MTEB calls encode() synchronously, but we may already be inside an
    event loop (via asyncio.run in the eval runner). We use a dedicated
    thread+loop to avoid nested asyncio.run() crashes.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    def encode(self, sentences: list[str], batch_size: int = 32, **kwargs: Any) -> Any:
        import concurrent.futures
        import numpy as np

        def _run_in_new_loop():
            loop = asyncio.new_event_loop()
            try:
                all_embeddings: list[list[float]] = []
                for i in range(0, len(sentences), batch_size):
                    batch = sentences[i:i + batch_size]
                    embeddings = loop.run_until_complete(self._client.embed_batch(batch))
                    all_embeddings.extend(embeddings)
                return all_embeddings
            finally:
                loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run_in_new_loop)
            all_embeddings = future.result()

        return np.array(all_embeddings)


def _mteb_results_to_bundle(task_name: str, results: list) -> dict[str, Any]:
    """Convert MTEB TaskResult list into NEL artifact bundle format."""
    scores: dict[str, Any] = {}
    all_metrics: dict[str, Any] = {}

    for task_result in results:
        for split_name, split_scores in task_result.scores.items():
            for score_dict in split_scores:
                for metric_name, value in score_dict.items():
                    if isinstance(value, (int, float)):
                        key = f"{split_name}/{metric_name}"
                        scores[key] = {"value": round(float(value), 4)}
                        all_metrics[key] = float(value)

    main_score = None
    for task_result in results:
        if hasattr(task_result, "main_score"):
            main_score = task_result.main_score

    if main_score and f"test/{main_score}" in scores:
        scores["main"] = scores[f"test/{main_score}"]

    return {
        "benchmark": {
            "name": f"mteb/{task_name}",
            "samples": 0,
            "scores": scores,
        },
        "config": {"benchmark": f"mteb/{task_name}", "framework": "mteb"},
        "_mteb_raw": json.loads(json.dumps(all_metrics)),
    }
