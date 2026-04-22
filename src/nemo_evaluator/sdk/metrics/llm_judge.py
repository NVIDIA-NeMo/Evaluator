"""LLM judge metric runtime implementation."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypeVar

import nemo_evaluator.sdk.inference as inference
from nemo_evaluator.sdk.enums import ModelFormat
from nemo_evaluator.sdk.inference import InferenceFn, InferenceHookParams
from nemo_evaluator.sdk.inference import new_hooks as _new_inference_hooks
from nemo_evaluator.sdk.metrics.hooks import HooksBase
from nemo_evaluator.sdk.metrics.template_rendering import build_template_context
from nemo_evaluator.sdk.structured_output import InferenceStructuredOutput, detect_structured_output_mode
from nemo_evaluator.sdk.templates import render_request
from nemo_evaluator.sdk.values.common import SecretRef, SupportedJobTypes
from nemo_evaluator.sdk.values.metrics import (
    LLMJudge,
    default_judge_prompt_template_chat,
    default_judge_prompt_template_completions,
    is_chat_inference,
)
from nemo_evaluator.sdk.values.models import Model
from nemo_evaluator.sdk.values.params import InferenceParams, ReasoningParams
from nemo_evaluator.sdk.values.results import MetricResult, MetricScore
from nemo_evaluator.sdk.values.scores import (
    JSONScoreParser,
    RangeScore,
    RubricScore,
    Score,
    ScoreParser,
    ScoreParserJSON,
    ScoreParserRegex,
)
from pydantic import PrivateAttr

__all__ = [
    "InferenceParams",
    "LLMJudgeMetric",
    "Model",
    "ReasoningParams",
    "Score",
    "generate_structured_output",
    "new_hooks",
]

_logger = logging.getLogger(__name__)
_T = TypeVar("_T", float, bool, MetricResult)


class _LLMJudgeHookParams(InferenceHookParams, Protocol):
    model: Model
    scores: list[Score]
    prompt_template: str | dict


class LLMJudgeMetric(HooksBase, LLMJudge):
    """Runtime metric implementation for LLM-as-a-judge scoring."""

    metric_threshold: float | None = None
    _metric_threshold_score: str | None = None
    _use_max_completion_tokens: bool = False
    _api_key: str | None = None
    _inference_fn: InferenceFn | None = None
    _parsers: dict[str, ScoreParser] = PrivateAttr(default_factory=dict)
    _score_dumps: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    job_type: Literal[SupportedJobTypes.ONLINE, SupportedJobTypes.OFFLINE] = SupportedJobTypes.ONLINE

    def set_inference_fn(self, inference_fn: InferenceFn) -> None:
        """Set the inference function to use for LLM calls."""
        self._inference_fn = inference_fn

    def score_names(self) -> list[str]:
        """Return score keys emitted by this metric."""
        return list(self._parsers.keys())

    def _handle_none_output_error(self, response: dict) -> ValueError:
        error_message = "LLM judge returned no usable textual content for score parsing"
        message = response.get("choices", [{}])[0].get("message")
        if isinstance(message, dict):
            has_reasoning = any(
                isinstance(value, str) and bool(value.strip())
                for value in (message.get("reasoning"), message.get("reasoning_content"))
            )
            if has_reasoning:
                error_message = (
                    f"{error_message}. The response contains reasoning output but no final text content; "
                    "the `max_tokens` budget may have been used entirely by reasoning. "
                    "Try increasing inference `max_tokens` "
                    "or configuring `inference.extra_body.nvext.max_thinking_tokens` for NIM endpoints"
                )
        return ValueError(f"{error_message}. Response: {response}.")

    def _validate_output_text(self, output_text: str | None, response: dict) -> str:
        """Ensure the judge returned textual content that can be parsed."""
        if isinstance(output_text, str):
            return output_text
        raise self._handle_none_output_error(response)

    def _handle_invalid_output(self, error: Exception, fallback: _T, message: str) -> _T:
        if self.ignore_inference_failure:
            _logger.warning("%s: %s", message, str(error))
            return fallback
        raise error

    def _nan_result(self) -> MetricResult:
        return MetricResult(scores=[MetricScore(name=name, value=float("nan")) for name in self._parsers])

    async def resolve_secrets(self, secret_resolver: Callable[[str], Awaitable[str | None]]) -> None:
        """Resolve API key secret if configured. Must be called before using the metric."""
        if self.model.api_key_secret:
            secret_name = self.model.api_key_secret.root
            self._api_key = await secret_resolver(secret_name)
            if not self._api_key:
                raise ValueError(f"Missing secret '{secret_name}' for API key authentication with LLM judge.")

    async def preflight(self) -> None:
        """Resolve structured-output mode once before parallel inference starts."""
        if self.model.format != ModelFormat.NVIDIA_NIM or not self.structured_output:
            return

        structured_hook: InferenceStructuredOutput | None = None
        for hook in self._preprocess_hooks:
            if isinstance(hook, InferenceStructuredOutput):
                structured_hook = hook
                break

        if structured_hook is None:
            return

        mode = await detect_structured_output_mode(
            format=self.model.format,
            model=self.model,
            inference_fn=self.inference_fn,
            api_key=self._api_key,
            probe_schema={
                "type": "object",
                "properties": {"__nmp_probe_score": {"type": "integer"}},
                "required": ["__nmp_probe_score"],
                "additionalProperties": False,
            },
        )
        structured_hook.set_mode(mode)
        _logger.info("NIM structured output mode selected: %s", mode.value)

    def secrets(self) -> dict[str, SecretRef]:
        """Return secret env mappings required by this metric."""
        if self.model.api_key_secret and self.model.api_key_env:
            return {self.model.api_key_env: self.model.api_key_secret}
        return {}

    @property
    def inference_fn(self) -> InferenceFn:
        """Get the inference function, defaulting to the global one if not injected."""
        return self._inference_fn or inference.make_inference_request

    def model_post_init(self, context: Any, /) -> None:
        # Pydantic runs model_post_init() during BaseModel construction, before any
        # custom __init__ logic would execute. Derive structured_output here so the
        # first parser initialization validates against the finalized JSON schema.
        self.structured_output = generate_structured_output(self)
        self._initialize_score_parsers()
        preprocess_hooks, postprocess_hooks = new_hooks(self)
        self.with_hooks(preprocess=preprocess_hooks, postprocess=postprocess_hooks)

        # Offline jobs cannot use sample.output_text default prompt; replace with item default.
        if "prompt_template" not in self.model_fields_set and self.job_type == SupportedJobTypes.OFFLINE:
            if is_chat_inference(self.model.url):
                self.prompt_template = default_judge_prompt_template_chat(self.job_type)
            else:
                self.prompt_template = default_judge_prompt_template_completions(self.job_type)
        return super().model_post_init(context)

    def _initialize_score_parsers(self) -> None:
        if not self.scores:
            return

        for score in self.scores:
            if not score.parser:
                raise ValueError(f"parser is required for LLM-as-a-Judge score {score.name}: {score}")

            parser_type = score.parser.type
            if parser_type == ScoreParserJSON.parser_type:
                parser = ScoreParserJSON(score=score, structured_output=self.structured_output)
            elif parser_type == ScoreParserRegex.parser_type:
                parser = ScoreParserRegex(score=score)
            else:
                raise ValueError(f"unknown parser type for LLM-as-a-Judge score {score.name}: {parser_type}")

            self._parsers[score.name] = parser
            self._score_dumps[score.name] = score.model_dump(mode="json", exclude={"parser"})

    @property
    def metric_threshold_score(self) -> str | None:
        return self._metric_threshold_score

    @metric_threshold_score.setter
    def metric_threshold_score(self, score_name: str):
        if score_name not in self._parsers:
            raise ValueError(f"score name does not match {self.type.value} scores {self._parsers.keys()}: {score_name}")
        self._metric_threshold_score = score_name

    def _render_request(self, item: dict, sample: dict) -> dict:
        overlapping_keys = set(item.keys()) & set(sample.keys())
        if overlapping_keys:
            _logger.warning(
                "Dataset columns %s overlap with model response keys. "
                "Model response values will be used. "
                "To access your dataset values, use 'item.<column_name>' in your template.",
                overlapping_keys,
            )

        context = build_template_context(item, sample)
        if self._score_dumps:
            context["scores"] = self._score_dumps
        request = render_request(self.prompt_template, context=context)

        if "max_tokens" not in request:
            request["max_tokens"] = 1024
        if self._use_max_completion_tokens:
            request["max_completion_tokens"] = request["max_tokens"]
            del request["max_tokens"]

        return self._apply_preprocess_hooks(request)

    def _retry_with_max_completion_tokens(self, request: dict) -> dict:
        if not self._use_max_completion_tokens:
            _logger.warning(
                "Model does not support 'max_tokens' parameter. Switching to 'max_completion_tokens' for all future requests."
            )
            self._use_max_completion_tokens = True
        request["max_completion_tokens"] = request["max_tokens"]
        del request["max_tokens"]
        return request

    def metric(self, item: dict, sample: dict, trace=None) -> float | bool:
        """Compute a single raw score for the metric."""
        if not (self.metric_threshold and self._metric_threshold_score):
            raise ValueError(
                f"metric_threshold and metric_threshold_score are required to compute metric for {self.type.value}"
            )

        request = self._render_request(item, sample)

        async def _invoke(request_data: dict[str, Any]) -> dict:
            return await self.inference_fn(self.model, request_data, 3, self._api_key)

        try:
            response = asyncio.run(_invoke(request))
        except inference.ClientInferenceError as error:
            if "max_tokens" in request and "'max_tokens' is not supported with this model" in error.args[0]:
                request = self._retry_with_max_completion_tokens(request)
                response = asyncio.run(_invoke(request))
            else:
                return self._handle_invalid_output(
                    error,
                    float("nan"),
                    "Inference failed with LLM judge, marking as NaN",
                )

        try:
            output_text = self._validate_output_text(
                inference.process_output(response, hooks=self._postprocess_hooks),
                response,
            )
        except ValueError as error:
            fallback: bool | float = False if trace else float("nan")
            message = (
                "LLM judge returned invalid output, marking as failed"
                if trace
                else "LLM judge returned invalid output, marking as NaN"
            )
            return self._handle_invalid_output(error, fallback, message)

        assert self.metric_threshold_score is not None
        parser = self._parsers[self.metric_threshold_score]
        score = parser.parse(output_text)
        _logger.debug("Parsed score %s: %s", self.metric_threshold_score, score.value)

        if trace:
            return score.value >= self.metric_threshold
        return score.value

    async def compute_scores(self, item: dict, sample: dict) -> MetricResult:
        """Compute structured score output for one item/sample pair."""
        request = self._render_request(item, sample)

        try:
            response = await self.inference_fn(self.model, request, 3, self._api_key)
        except inference.ClientInferenceError as error:
            if "max_tokens" in request and "'max_tokens' is not supported with this model" in error.args[0]:
                request = self._retry_with_max_completion_tokens(request)
                response = await self.inference_fn(self.model, request, 3, self._api_key)
            else:
                return self._handle_invalid_output(
                    error,
                    self._nan_result(),
                    "Inference failed with LLM judge, marking as NaN",
                )

        try:
            output_text = self._validate_output_text(
                inference.process_output(response, hooks=self._postprocess_hooks),
                response,
            )
        except ValueError as error:
            return self._handle_invalid_output(
                error,
                self._nan_result(),
                "LLM judge returned invalid output, marking as NaN",
            )

        result = MetricResult(scores=[])
        for score_name, parser in self._parsers.items():
            score = parser.parse(output_text)
            _logger.debug("Parsed score %s: %s", score_name, score.value)
            result.scores.append(score)
        return result


def new_hooks(params: _LLMJudgeHookParams | None):
    """Initialize preprocess and postprocess hooks for the LLM judge."""
    model_format = params.model.format if params else ModelFormat.NVIDIA_NIM
    return _new_inference_hooks(params, model_format=model_format)


def generate_structured_output(params: _LLMJudgeHookParams) -> dict | None:
    """Derive JSON schema for LLM structured output from score criteria."""
    if params.structured_output:
        return params.structured_output

    properties: dict[str, dict[str, Any]] = {}
    for score in params.scores:
        if not isinstance(score.parser, JSONScoreParser):
            continue

        key = score.parser.json_path
        if isinstance(score, RubricScore):
            schema = {
                "type": "string",
                "enum": [rubric.label for rubric in score.rubric],
            }
        elif isinstance(score, RangeScore):
            schema = {
                "type": "integer" if isinstance(score.minimum, int) else "number",
                "minimum": score.minimum,
                "maximum": score.maximum,
            }
        else:
            continue

        existing = properties.get(key)
        if existing is not None and existing != schema:
            raise ValueError(
                f"conflicting auto-generated structured_output for json_path '{key}'; "
                "provide explicit structured_output"
            )
        properties[key] = schema

    if not properties:
        return None

    return {
        "schema": {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
        }
    }
