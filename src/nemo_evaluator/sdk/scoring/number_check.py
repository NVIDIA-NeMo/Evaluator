"""Number-check metric runtime implementation."""

import math
import re

from nemo_evaluator.sdk.scoring.base import NELScorerMixin
from nemo_evaluator.sdk.scoring.template_rendering import (
    build_template_context,
    render_template_or_raise,
    template_metric_repr,
)
from nemo_evaluator.sdk.values.metrics import NumberCheck, NumberCheckOperation
from nemo_evaluator.sdk.values.results import MetricResult, MetricScore

__all__ = ["NumberCheckMetric", "NumberCheckOperation"]


def _parse_number_answer(answer: str) -> int | float:
    """Parse the last numeric value from text; return ``NaN`` when parsing fails."""
    numbers = re.findall(r"[+-]?[\.\d]*\d+", answer)
    if not numbers:
        return float("nan")

    last_number = numbers[-1]
    decimal = last_number.count(".")
    if decimal == 1:
        return float(last_number)
    if decimal == 0:
        return int(last_number)
    return float("nan")


class NumberCheckMetric(NELScorerMixin, NumberCheck):
    """Numeric-comparison metric with template-driven operands."""

    def score_names(self) -> list[str]:
        """Return score keys emitted by this metric."""
        return [self.type.value]

    def metric(self, item: dict, sample: dict, trace=None) -> float | bool:
        """Compute a single raw score for the metric."""
        context = build_template_context(item, sample)
        metric_repr = template_metric_repr(self)
        left_value = render_template_or_raise(
            template_name="left_template",
            template=self.left_template,
            context=context,
            item=item,
            sample=sample,
            metric_repr=metric_repr,
        )
        right_value = render_template_or_raise(
            template_name="right_template",
            template=self.right_template,
            context=context,
            item=item,
            sample=sample,
            metric_repr=metric_repr,
        )

        left_number = _parse_number_answer(str(left_value))
        right_number = _parse_number_answer(str(right_value))
        # Preserve the legacy behavior: if either side fails to parse as a
        # number, return NaN instead of raising.
        if math.isnan(left_number):
            return left_number
        if math.isnan(right_number):
            return right_number

        # Perform the requested numeric comparison on the parsed operands.
        if self.operation in ["equals", "=="]:
            score = left_number == right_number
        elif self.operation in ["!=", "<>", "not equals"]:
            score = left_number != right_number
        elif self.operation in [">=", "gte", "greater than or equal"]:
            score = left_number >= right_number
        elif self.operation in [">", "gt", "greater than"]:
            score = left_number > right_number
        elif self.operation in ["<=", "lte", "less than or equal"]:
            score = left_number <= right_number
        elif self.operation in ["<", "lt", "less than"]:
            score = left_number < right_number
        elif self.operation == "absolute difference":
            if self.epsilon is None:
                raise ValueError("epsilon value is required with operation absolute difference")
            score = abs(left_number - right_number) <= self.epsilon
        else:
            raise ValueError(f"Unsupported operation: {self.operation}")

        if trace:
            return score
        return 1.0 if score else 0.0

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute the scores for the metric."""
        score = self.metric(item, sample)
        return MetricResult(scores=[MetricScore(name=self.type.value, value=score)])
