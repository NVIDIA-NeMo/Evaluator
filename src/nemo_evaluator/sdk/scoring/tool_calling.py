"""Tool-calling metric runtime implementation."""

import json
import logging
from typing import ClassVar, Literal

from nemo_evaluator.sdk.scoring.template_rendering import (
    build_template_context,
    render_template_or_raise,
    template_metric_repr,
)
from nemo_evaluator.sdk.values.metrics import ToolCalling
from nemo_evaluator.sdk.values.results import MetricResult, MetricScore

__all__ = ["ToolCallingMetric"]

_logger = logging.getLogger(__name__)


class ToolCallingMetric(ToolCalling):
    """Tool-calling accuracy metric for structured function calls.

    A metric that supports checks of tool calling:
    - function names
    - function names and args
    and produces respective scores.

    Important. This metric:
    - is case sensitive (for all scores)
    - is order insensitive, so parallel multiple tool calls may appear out of order
    - expects the ground truth to be formatted in OpenAI-compliant tool calling format
    - requires function names with dots (``.``) to be normalized to underscores
      (``_``), since dots are not supported in OpenAI function names

    Evaluator-driven runs should usually source ``reference`` from ``item``.
    """

    metric_threshold: float | None = None

    _score_names: ClassVar[list[str]] = ["function_name_accuracy", "function_name_and_args_accuracy"]
    _metric_threshold_score: Literal["function_name_accuracy", "function_name_and_args_accuracy"] | None = None

    def score_names(self) -> list[str]:
        """Return score keys emitted by this metric."""
        return list(self._score_names)

    @property
    def metric_threshold_score(self):
        """The score used to apply metric threshold when multiple scores are computed."""
        return self._metric_threshold_score

    @metric_threshold_score.setter
    def metric_threshold_score(self, score_name: Literal["function_name_accuracy", "function_name_and_args_accuracy"]):
        """Set the score used to apply metric threshold when multiple scores are computed."""
        if score_name not in self._score_names:
            raise ValueError(f"score name does not match {self.type.value} scores {self._score_names}: {score_name}")
        self._metric_threshold_score = score_name

    def _metric(self, item: dict, sample: dict) -> dict[str, float]:
        """Compute raw tool-calling scores for one item/sample pair."""
        context = build_template_context(item, sample)
        ground_truth = render_template_or_raise(
            template_name="reference",
            template=self.reference,
            context=context,
            item=item,
            sample=sample,
            metric_repr=template_metric_repr(self),
            item_keys_label="reference",
            sample_keys_label="sample",
        )
        if not isinstance(ground_truth, list):
            raise TypeError("The reference must render to a list of OpenAI-style tool calls.")

        try:
            gt_fn_names = [gt["function"]["name"] for gt in ground_truth]
            gt_fn_args_by_name = {gt["function"]["name"]: gt["function"]["arguments"] for gt in ground_truth}
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Invalid reference template - expected each item to have function.name and function.arguments: {e}"
            ) from e

        # Parse tool calls: check sample (online) first, then item (offline).
        response_data = sample.get("response") or item.get("response")
        if not response_data:
            raise ValueError("No response found in sample or item - tool-calling metric requires model response data")
        if not isinstance(response_data, dict):
            raise ValueError(f"Invalid response format: expected response dict, got {response_data!r}.")

        choices = response_data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"Invalid response format: expected non-empty choices list in response {response_data!r}.")

        first_choice = choices[0]
        if not isinstance(first_choice, dict) or "message" not in first_choice:
            raise ValueError(f"Invalid response format: expected choices[0].message in response {response_data!r}.")

        message = first_choice["message"]

        if not message.get("tool_calls"):
            _logger.info("No tool calls found in %s", sample)
            message["tool_calls"] = []

        tool_calls = message["tool_calls"]
        pred_fn_name = [call["function"]["name"] for call in tool_calls if "name" in call.get("function", {})]

        fn_names_match = set(gt_fn_names) == set(pred_fn_name)
        fn_name_accuracy_score = 1.0 if fn_names_match else 0.0
        fn_name_and_args_accuracy_score = 0.0

        try:
            pred_fn_args_by_name: dict[str, dict] = {}
            for pred in tool_calls:
                function_payload = pred.get("function", {})
                if "name" not in function_payload:
                    continue

                args = function_payload.get("arguments")
                if args is None:
                    parsed_args = {}
                else:
                    parsed_args = json.loads(args)

                pred_fn_args_by_name[function_payload["name"]] = parsed_args

            _logger.debug("Comparing %s and %s", gt_fn_args_by_name, pred_fn_args_by_name)

            all_args_match = True
            if not fn_names_match:
                # If function names do not match, the combined name-and-args
                # score must also fail automatically.
                fn_name_and_args_accuracy_score = 0.0
            else:
                # Compare parsed arguments for each ground-truth function name.
                for gt_fn_name, gt_fn_args in gt_fn_args_by_name.items():
                    pred_fn_args = pred_fn_args_by_name.get(gt_fn_name)

                    if pred_fn_args != gt_fn_args:
                        all_args_match = False
                        break

                fn_name_and_args_accuracy_score = 1.0 if all_args_match else 0.0
        except (json.JSONDecodeError, TypeError):
            # If the model hallucinated malformed JSON arguments, preserve the
            # legacy behavior and report NaN for the args-sensitive score.
            _logger.warning("Failed parsing tool calling function args: %s", tool_calls)
            fn_name_and_args_accuracy_score = float("nan")

        return {
            "function_name_accuracy": fn_name_accuracy_score,
            "function_name_and_args_accuracy": fn_name_and_args_accuracy_score,
        }

    def metric(self, item: dict, sample: dict, trace=None) -> float | bool:
        """Compute a single raw score for the metric."""
        if not (self.metric_threshold and self.metric_threshold_score):
            raise ValueError(
                f"metric_threshold and metric_threshold_score are required to compute metric for {self.type.value}"
            )

        scores = self._metric(item, sample)
        score = scores[self.metric_threshold_score]

        if trace:
            return score >= self.metric_threshold

        return score

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute the scores for the metric."""
        scores = self._metric(item, sample)
        return MetricResult(scores=[MetricScore(name=score_name, value=score) for score_name, score in scores.items()])
