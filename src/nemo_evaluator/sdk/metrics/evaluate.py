"""Pure evaluation functions for metric computation.

This module contains evaluation logic that is independent of storage/service concerns.
It can be used by both the live API endpoint and the job system.
"""

# Migrated from: services/evaluator/src/nmp/evaluator/app/metrics/evaluate.py

import asyncio
import logging
from typing import Any

from nemo_evaluator.sdk.metrics.base import Metric
from nemo_evaluator.sdk.values.results import SampleResult

_logger = logging.getLogger(__name__)


class EvaluationError(Exception):
    """Raised when evaluation fails in fail_fast mode."""

    def __init__(self, index: int, message: str):
        """Create an evaluation error that keeps sample index context.

        Args:
            index: Position of the failing sample in the input list.
            message: Original exception message.
        """
        super().__init__(message)
        self.index = index
        self.message = message


async def evaluate_samples(
    metric: Metric,
    samples: list[dict[str, Any]],
    parallelism: int = 8,
    fail_fast: bool = False,
) -> list[SampleResult]:
    """
    Evaluate samples against a metric.

    This is a pure evaluation function with no dependencies on storage or services.
    It can be easily tested by passing a mock metric.

    Args:
        metric: An instantiated metric (from new_metric()).
        samples: List of row dicts to evaluate. Each dict is passed as the
            item argument to the metric's compute_scores method.
        parallelism: Maximum number of concurrent evaluations.
        fail_fast: If True, raise EvaluationError on first failure instead of collecting errors.
            Note: In fail_fast mode, evaluation is still parallel but cancels on first failure.

    Returns:
        List of SampleResult, one per sample, sorted by index.
        Each result contains either the metric scores or an error message.

    Raises:
        EvaluationError: If fail_fast=True and any evaluation fails.
    """
    if not samples:
        return []

    semaphore = asyncio.Semaphore(parallelism)

    async def evaluate_one_strict(index: int, row: dict[str, Any]) -> SampleResult:
        """Evaluate one row and propagate errors to the caller.

        Args:
            index: Row index in ``samples``.
            row: Row payload to evaluate.

        Returns:
            Successful sample result for this row.

        Raises:
            EvaluationError: If metric execution fails for this row.
        """
        async with semaphore:
            try:
                result = await metric.compute_scores(row, {})
                return SampleResult.success(index, result)
            except Exception as e:
                raise EvaluationError(index, str(e)) from e

    async def evaluate_one(index: int, row: dict[str, Any]) -> SampleResult:
        """Collects failures as SampleResult."""
        async with semaphore:
            try:
                result = await metric.compute_scores(row, {})
                return SampleResult.success(index, result)
            except Exception as e:
                _logger.warning("Failed to evaluate row %d: %s", index, e)
                return SampleResult.failure(index, e)

    evaluate_fn = evaluate_one_strict if fail_fast else evaluate_one
    tasks = [asyncio.create_task(evaluate_fn(i, row)) for i, row in enumerate(samples)]

    if fail_fast:
        # FIRST_EXCEPTION gives strict callers the original failure instead of
        # waiting for unrelated work to finish in the background.
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if (exc := task.exception()) is not None:
                raise exc
    else:
        done, _ = await asyncio.wait(tasks)

    # Preserve input ordering even though tasks finish out of order.
    return sorted([t.result() for t in done], key=lambda r: r.index)
